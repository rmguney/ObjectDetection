import subprocess

def main():
    print("Select an option:")
    print("1. Run Classification Model Comparison")
    print("2. Run Detection Model Comparison")
    print("3. Launch GUI for Manual Testing")
    print("4. Train MobileNet")
    print("5. Train DETR")
    print("6. Test MobileNet")
    print("7. Test DETR")
    print("8. Test Dataset")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        # Run classification model comparison script
        from src.compare_models_classification import main as compare_models_classification
        epochs = int(input("Enter number of epochs for classification model comparison: "))
        compare_models_classification(epochs=epochs)
    elif choice == "2":
        # Run detection model comparison script
        from src.compare_models import main as compare_models
        epochs = int(input("Enter number of epochs for detection model comparison: "))
        compare_models(epochs=epochs)
    elif choice == "3":
        # Launch GUI for manual testing
        subprocess.run(["python", "src/gui_test.py"])
    elif choice == "4": 
        from src.train_mobilenet import train_mobilenet
        epochs = int(input("Enter number of epochs for MobileNet SSD: "))
        train_mobilenet(epochs=epochs)
    elif choice == "5":
        from src.train_detr import train_detr
        epochs = int(input("Enter number of epochs for DETR: "))
        train_detr(epochs=epochs)
    elif choice == "6":
        from src.test_mobilenet import test_mobilenet
        test_mobilenet()
    elif choice == "7":
        from src.test_detr import test_detr
        test_detr()
    elif choice == "8":
        from src.test_dataset import test_dataset
        test_dataset()
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
