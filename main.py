import subprocess

def main():
    print("Select an option:")
    print("1. Train MobileNet SSD")
    print("2. Train DETR")
    print("3. Test MobileNet SSD")
    print("4. Test DETR")
    print("5. Test Dataset")
    print("6. Launch GUI for Manual Testing")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        from src.train_mobilenet import train_mobilenet
        epochs = int(input("Enter number of epochs for MobileNet SSD: "))
        train_mobilenet(epochs=epochs)
    elif choice == "2":
        from src.train_detr import train_detr
        epochs = int(input("Enter number of epochs for DETR: "))
        train_detr(epochs=epochs)
    elif choice == "3":
        from src.test_mobilenet import test_mobilenet
        test_mobilenet()
    elif choice == "4":
        from src.test_detr import test_detr
        test_detr()
    elif choice == "5":
        from src.test_dataset import test_dataset
        test_dataset()
    elif choice == "6":
        # Launch GUI for manual testing
        subprocess.run(["python", "src/gui_test.py"])
    else:
        print("Invalid choice. Please select 1, 2, 3, 4, 5, or 6.")

if __name__ == "__main__":
    main()