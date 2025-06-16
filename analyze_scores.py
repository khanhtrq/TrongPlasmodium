import csv
from sklearn.metrics import classification_report, accuracy_score

def calculate_classification_metrics_v2(file_path):
    """
    Đọc file dữ liệu, trích xuất nhãn thật và nhãn dự đoán,
    sau đó tính toán và in ra các chỉ số P, R, F1, accuracy tổng thể,
    và macro accuracy (trung bình accuracy của mỗi lớp).

    Args:
        file_path (str): Đường dẫn đến file .txt chứa dữ liệu.
                         Định dạng mỗi dòng: <đường_dẫn_ảnh>,<label_thật>,<label_dự_đoán>,...
    """
    true_labels = []
    predicted_labels = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    true_labels.append(row[1])
                    predicted_labels.append(row[2])
                else:
                    print(f"Bỏ qua dòng không đủ cột: {row}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    if not true_labels or not predicted_labels:
        print("Không có dữ liệu nhãn nào được tìm thấy trong file.")
        return

    unique_labels_sorted = sorted(list(set(true_labels + predicted_labels)))

    print("Báo cáo phân loại chi tiết:\n")
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=unique_labels_sorted,
        zero_division=0,
        digits=4
    )
    print(report)

    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy tổng thể (Overall Accuracy): {overall_accuracy:.4f}")

    # Tính toán Macro Accuracy (trung bình accuracy của mỗi lớp)
    # Chỉ tính cho 4 lớp đầu tiên
    first_4_classes = unique_labels_sorted[:4]
    
    # Lọc dữ liệu chỉ giữ lại các mẫu thuộc 4 lớp đầu
    filtered_true_labels = []
    filtered_predicted_labels = []
    
    for i in range(len(true_labels)):
        if true_labels[i] in first_4_classes:
            filtered_true_labels.append(true_labels[i])
            filtered_predicted_labels.append(predicted_labels[i])
    
    filtered_total_samples = len(filtered_true_labels)
    per_class_accuracies = []

    if filtered_total_samples == 0:
        print(f"\nKhông có mẫu nào thuộc 4 lớp đầu ({first_4_classes}) để tính toán macro accuracy.")
    else:
        print(f"\n--- Chi tiết tính toán Macro Accuracy cho 4 lớp đầu: {first_4_classes} ---")
        print(f"Số mẫu được sử dụng: {filtered_total_samples}/{len(true_labels)}")
        
        for current_class in first_4_classes:
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for i in range(filtered_total_samples):
                true_label = filtered_true_labels[i]
                predicted_label = filtered_predicted_labels[i]

                if true_label == current_class and predicted_label == current_class:
                    tp += 1
                elif true_label != current_class and predicted_label == current_class:
                    fp += 1
                elif true_label == current_class and predicted_label != current_class:
                    fn += 1
                elif true_label != current_class and predicted_label != current_class:
                    tn += 1
            
            # Accuracy cho lớp hiện tại = (TP + TN) / Tổng số mẫu được lọc
            class_accuracy = (tp + tn) / filtered_total_samples if filtered_total_samples > 0 else 0
            per_class_accuracies.append(class_accuracy)
            
            print(f"  Lớp '{current_class}':")
            print(f"    TP (True Positives) = {tp}")
            print(f"    TN (True Negatives) = {tn}")
            print(f"    FP (False Positives) = {fp}")
            print(f"    FN (False Negatives) = {fn}")
            print(f"    Accuracy lớp '{current_class}' = (TP + TN) / Tổng số mẫu được lọc = ({tp} + {tn}) / {filtered_total_samples} = {class_accuracy:.4f}")

        if per_class_accuracies:
            macro_accuracy = sum(per_class_accuracies) / len(per_class_accuracies)
            print(f"\nMacro Accuracy cho 4 lớp đầu (trung bình các accuracy): {macro_accuracy:.4f}")
        else:
            print(f"\nKhông có lớp nào trong 4 lớp đầu để tính macro accuracy.")
            
    print(f"\nGiải thích các chỉ số từ classification_report:")
    print(f"- Macro Average P, R, F1: Trung bình cộng không trọng số của P, R, F1 của từng lớp.")
    print(f"- Weighted Average P, R, F1: Trung bình có trọng số (theo số lượng mẫu của mỗi lớp) của P, R, F1 của từng lớp.")


if __name__ == "__main__":
    # Thay 'du_lieu_cua_ban.txt' bằng đường dẫn thực tế đến file của bạn
    file_path = r"X:\datn\finetune\bmc wd maxnorm\PlasmodiumClassification-1\results_kaggle\efficientnet_b1.ra4_e3600_r240_in1k_classifier_finetune\efficientnet_b1.ra4_e3600_r240_in1k_classifier_test_eval_predictions.txt"
    # Nếu bạn muốn chạy với file của mình:
    print(f"\n--- Phân tích file: {file_path} ---")
    calculate_classification_metrics_v2(file_path)