import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def plot_confusion_matrix_manual(y_true, y_pred, class_names, title='Confusion Matrix',
                                 normalize=True, figsize=(10, 8), save_path=None):
    """
    Vẽ confusion matrix thủ công với khả năng tùy chỉnh cao

    Args:
        y_true: Nhãn thật
        y_pred: Nhãn dự đoán  
        class_names: Danh sách tên các lớp
        title: Tiêu đề của biểu đồ
        normalize: True để chuẩn hóa về tỷ lệ phần trăm
        figsize: Kích thước figure
        save_path: Đường dẫn lưu ảnh (None = không lưu)
    """
    # Mapping class names to custom labels
    label_mapping = {
        '0': 'TJ',
        '1': 'TA',
        '2': 'S',
        '3': 'G',
        '4': 'UN'
    }

    # Convert class names to custom labels
    custom_labels = []
    for class_name in class_names:
        # Tìm mapping phù hợp (case insensitive)
        mapped_label = None
        for key, value in label_mapping.items():
            if key.lower() in class_name.lower():
                mapped_label = value
                break
        # Nếu không tìm thấy mapping, giữ nguyên tên gốc
        custom_labels.append(mapped_label if mapped_label else class_name)

    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Xử lý chia cho 0
        display_cm = cm_normalized
        fmt = '.2%'
        title += ' (Normalized)'
    else:
        display_cm = cm
        fmt = 'd'

    # Tính toán độ dài title tối đa dựa trên kích thước matrix
    # Ít nhất 40 ký tự, tăng theo số lớp
    max_chars_per_line = max(40, len(class_names) * 8)

    # Chia title thành nhiều dòng nếu quá dài
    def wrap_title(title_text, max_length):
        words = title_text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line = current_line + " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    wrapped_title = wrap_title(title, max_chars_per_line)
    num_title_lines = len(wrapped_title.split('\n'))

    # Tạo figure và axis với borders động dựa trên số dòng title
    fig, ax = plt.subplots(figsize=figsize)

    # Điều chỉnh top margin dựa trên số dòng title
    top_margin = max(0.85, 0.95 - (num_title_lines - 1) * 0.05)
    plt.subplots_adjust(top=top_margin, bottom=0.2, left=0.15, right=0.85)

    # Vẽ heatmap thủ công
    im = ax.imshow(display_cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Thêm colorbar với padding
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    if normalize:
        cbar.ax.set_ylabel('Tỷ lệ phần trăm', rotation=-90,
                           va="bottom", fontsize=12, labelpad=20)
    else:
        cbar.ax.set_ylabel('Số lượng', rotation=-90,
                           va="bottom", fontsize=12, labelpad=20)

    # Cấu hình axes với custom labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=custom_labels,  # Sử dụng custom labels
           yticklabels=custom_labels)  # Sử dụng custom labels

    # Cải thiện font size và padding cho title đã wrap
    # Tăng padding cho title nhiều dòng
    title_pad = max(20, num_title_lines * 15)
    ax.set_title(wrapped_title, fontsize=16, fontweight='bold', pad=title_pad)
    ax.set_xlabel('Nhãn dự đoán (Predicted Label)', fontsize=14, labelpad=15)
    ax.set_ylabel('Nhãn thật (True Label)', fontsize=14, labelpad=15)

    # Xoay labels trên trục x với padding tốt hơn
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Thêm text annotations với font size phù hợp
    thresh = display_cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{display_cm[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{display_cm[i, j]:{fmt}}'

            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if display_cm[i, j] > thresh else "black",
                    fontsize=11, fontweight='bold')

    # Lưu file với tight bbox
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                    edgecolor='none', pad_inches=0.3)
        print(f"Đã lưu confusion matrix tại: {save_path}")

    plt.show()

    # In thống kê chi tiết với custom labels
    print(f"\n--- Thống kê Confusion Matrix ---")
    print(f"Tổng số mẫu: {np.sum(cm)}")
    print(f"Mapping labels: {dict(zip(class_names, custom_labels))}")

    for i, (class_name, custom_label) in enumerate(zip(class_names, custom_labels)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nLớp '{custom_label}' ({class_name}):")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(
            f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


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

    # Vẽ confusion matrix - bổ sung mới
    print(f"\n{'='*50}")
    print("VẼ CONFUSION MATRIX")
    print(f"{'='*50}")

    # Confusion matrix cho tất cả các lớp
    save_dir = file_path.replace('.txt', '_confusion_matrix_all.png')
    plot_confusion_matrix_manual(
        true_labels, predicted_labels, unique_labels_sorted,
        title='Confusion Matrix - WD + MaxNorm + Class Balanced Softmax Finetune',
        normalize=False,
        figsize=(12, 10),
        save_path=save_dir
    )

    # Confusion matrix chuẩn hóa
    save_dir_norm = file_path.replace(
        '.txt', '_confusion_matrix_normalized.png')
    plot_confusion_matrix_manual(
        true_labels, predicted_labels, unique_labels_sorted,
        title='Confusion Matrix - WD + MaxNorm + Class Balanced Softmax Finetune',
        normalize=True,
        figsize=(12, 10),
        save_path=save_dir_norm
    )

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
        print(
            f"\nKhông có mẫu nào thuộc 4 lớp đầu ({first_4_classes}) để tính toán macro accuracy.")
    else:
        print(
            f"\n--- Chi tiết tính toán Macro Accuracy cho 4 lớp đầu: {first_4_classes} ---")
        print(
            f"Số mẫu được sử dụng: {filtered_total_samples}/{len(true_labels)}")

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
            class_accuracy = (
                tp + tn) / filtered_total_samples if filtered_total_samples > 0 else 0
            per_class_accuracies.append(class_accuracy)

            print(f"  Lớp '{current_class}':")
            print(f"    TP (True Positives) = {tp}")
            print(f"    TN (True Negatives) = {tn}")
            print(f"    FP (False Positives) = {fp}")
            print(f"    FN (False Negatives) = {fn}")
            print(
                f"    Accuracy lớp '{current_class}' = (TP + TN) / Tổng số mẫu được lọc = ({tp} + {tn}) / {filtered_total_samples} = {class_accuracy:.4f}")

        if per_class_accuracies:
            macro_accuracy = sum(per_class_accuracies) / \
                len(per_class_accuracies)
            print(
                f"\nMacro Accuracy cho 4 lớp đầu (trung bình các accuracy): {macro_accuracy:.4f}")
        else:
            print(f"\nKhông có lớp nào trong 4 lớp đầu để tính macro accuracy.")

    print(f"\nGiải thích các chỉ số từ classification_report:")
    print(f"- Macro Average P, R, F1: Trung bình cộng không trọng số của P, R, F1 của từng lớp.")
    print(f"- Weighted Average P, R, F1: Trung bình có trọng số (theo số lượng mẫu của mỗi lớp) của P, R, F1 của từng lớp.")


if __name__ == "__main__":
    # Thay 'du_lieu_cua_ban.txt' bằng đường dẫn thực tế đến file của bạn
    file_path = r"X:\datn\new\new focal max\PlasmodiumClassification-1\results_kaggle\efficientnet_b1.ra4_e3600_r240_in1k_classifier_finetune\efficientnet_b1.ra4_e3600_r240_in1k_classifier_test_eval_predictions.txt"
    # Nếu bạn muốn chạy với file của mình:
    print(f"\n--- Phân tích file: {file_path} ---")
    calculate_classification_metrics_v2(file_path)
