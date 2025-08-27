import argparse

import numpy as np
from cs336_basics.model import BasicsTransformerLM
import torch
import timeit
import os

#TODO: Support more detailed profiling
def parse_options():
    # model options
    parser = argparse.ArgumentParser(description="Profiler Options")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads in the model")
    parser.add_argument("--d_ff", type=int, default=256, help="Dimension of the hidden layers in the model")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--rope_theta", type=float, default=100000.0, help="Rope theta value")
    parser.add_argument("--max_context_length", type=int, default=256, help="Maximum context length")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")

    # data options
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    # profiler options
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps for profiling")
    parser.add_argument("--forward_only", action='store_true' ,help="Profiling mode")
    parser.add_argument("--benchmark_steps", type=int, default=10, help="Number of benchmark steps for profiling")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--output_file", type=str, default="output.csv", help="Output file to save the results")


    args = parser.parse_args()
    return args


def _benchmark(model, data, warmup_steps, benchmark_steps, forward_only, device):
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # warmup
    print("Starting warmup...")
    for _ in range(warmup_steps):
        outputs = model(data)
        loss = outputs.sum()
        loss.backward()
        model.zero_grad()

    print("Starting benchmark...")
    # benchmark
    forward_time = []
    backward_time = []
    for _ in range(benchmark_steps):
        start_time = timeit.default_timer()
        outputs = model(data)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_time.append(timeit.default_timer() - start_time)

        if not forward_only:
            start_time = timeit.default_timer()
            optimizer.zero_grad()
            loss = outputs.sum()
            loss.backward()
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            backward_time.append(timeit.default_timer() - start_time)

    avg_forward_time = (sum(forward_time) / len(forward_time)) if forward_time else 'N/A'
    avg_backward_time = (sum(backward_time) / len(backward_time)) if backward_time else 'N/A'
    std_forward_time = np.std(forward_time) if forward_time else 'N/A'
    std_backward_time = np.std(backward_time) if backward_time else 'N/A'
    return avg_forward_time, avg_backward_time, std_forward_time, std_backward_time



def benchmark():
    args = parse_options()

    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        args.device = 'cuda' if torch.cuda.is_available() else args.device
        print(f"Using device: {args.device}")

    # initialize model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.max_context_length,
        d_model=args.d_model,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # random generate data
    input_ids_batch = torch.randint(0, args.vocab_size, (args.batch_size, args.max_context_length))
    avg_forward_time, avg_backward_time, std_forward_time, std_backward_time \
        = _benchmark(model, input_ids_batch, args.warmup_steps, args.benchmark_steps, args.forward_only, args.device)

    print(f"Average Forward Time: {avg_forward_time:.6f} seconds, Std: {std_forward_time:.6f} seconds")
    if not args.forward_only:
        print(f"Average Backward Time: {avg_backward_time:.6f} seconds, Std: {std_backward_time:.6f} seconds")

    # save results to output csv file
    os.makedirs("profiles", exist_ok=True)
    output_path = os.path.join("profiles", args.output_file)
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
                f.write("device, n_layers,n_heads,d_ff,d_model,max_context_length,vocab_size,batch_size,avg_forward_time,std_forward_time,avg_backward_time, std_backward_time\n")

    with open(output_path, "a") as f:
        f.write(f"{args.device},{args.n_layers},{args.n_heads},{args.d_ff},{args.d_model},{args.max_context_length},{args.vocab_size},"
                f"{args.batch_size},{avg_forward_time},{std_forward_time},{avg_backward_time},{std_backward_time}\n")


    return avg_forward_time, avg_backward_time

if __name__ == "__main__":
    benchmark()


