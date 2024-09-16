import torch

def main():
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.device_count()=}")
    print(f"{torch.cuda.current_device()=}")
    print(f"{torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
