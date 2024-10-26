import subprocess

def main():
    print("Select an option:")
    print("1. Run Model Comparison")
    print("2. Launch GUI for Manual Testing")
    print("3. Train MobileNet SSD")
    print("4. Train DETR")
    print("5. Test MobileNet SSD")
    print("6. Test DETR")
    print("7. Test Dataset")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        # Run model comparison script
        from src.compare_models import main as compare_models
        compare_models()
    elif choice == "2":
        # Launch GUI for manual testing
        subprocess.run(["python", "src/gui_test.py"])
    elif choice == "3": 
        from src.train_mobilenet import train_mobilenet
        epochs = int(input("Enter number of epochs for MobileNet SSD: "))
        train_mobilenet(epochs=epochs)
    elif choice == "4":
        from src.train_detr import train_detr
        epochs = int(input("Enter number of epochs for DETR: "))
        train_detr(epochs=epochs)
    elif choice == "5":
        from src.test_mobilenet import test_mobilenet
        test_mobilenet()
    elif choice == "6":
        from src.test_detr import test_detr
        test_detr()
    elif choice == "7":
        from src.test_dataset import test_dataset
        test_dataset()
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()