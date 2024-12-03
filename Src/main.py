import argparse
from single_thread import process_images_single
from parallel_comp import process_images_parallel

def main():
    parser = argparse.ArgumentParser(description="Image Processing Program")
    parser.add_argument("--mode", choices=["single", "parallel"], required=True,
                        help="Choose 'single' for single-threaded or 'parallel' for GPU processing.")
    args = parser.parse_args()

    if args.mode == "single":
        process_images_single()
    elif args.mode == "parallel":
        process_images_parallel()

if __name__ == "__main__":
    main()
