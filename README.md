
├── data/                                # Thư mục chứa dữ liệu (Dataset)
│  
├── notebook/                            # Thư mục chứa Jupyter Notebooks (Mã nguồn chính)
│   ├── Lab5_part1.pdf                    
│
├── README.md                           # File báo cáo chi tiết này
│
├── part1                                # Mã nguồn Python (Modules/Classes tái sử dụng)
│   
├── .gitignore                           # File cấu hình bỏ qua file rác (tmp, __pycache__)



1. Implementation Steps

Trong bài này, em thực hiện bài toán phân loại cảm xúc (sentiment classification) dựa trên tập dữ liệu gồm các câu bình luận đã được gán nhãn tích cực (1) hoặc tiêu cực (0). Các bước triển khai được thực hiện như sau:

Bước 1: Chuẩn bị và xử lý dữ liệu

  Tải dữ liệu từ sent_train.csv và sent_valid.csv.
  
  Chuẩn hóa nhãn về kiểu int.
  
  Loại bỏ các dòng bị thiếu dữ liệu.

Bước 2: Biểu diễn văn bản (Feature Extraction)
  
  Sử dụng TF-IDF Vectorizer để chuyển đổi văn bản thành vector đặc trưng.
  
  Thiết lập max_features=5000 để giảm số chiều và tránh overfitting.

Bước 3: Baseline Model
  
  Huấn luyện mô hình Logistic Regression trên tập train.
  
  Thực hiện dự đoán trên tập valid và đánh giá mô hình.

Bước 4: Mô hình trên Spark (Task 3)
  
  Sử dụng Spark ML Pipeline gồm:
    
    Tokenizer
    
    StopWordsRemover
    
    HashingTF
    
    IDF
    
    LogisticRegression
  
  Training và đánh giá mô hình trên toàn bộ dataset sentiments.csv.

Bước 5: Mô hình cải tiến (Task 4)
  
  Dùng TF-IDF với n-grams (1,2).
  
  Huấn luyện mô hình Multinomial Naive Bayes.
  
  So sánh với baseline và mô hình Spark

2. Code Execution Guide
  Chạy bằng Notebook Google Colab
  
  Mở file notebook lab5_spark_sentiment_analysis.ipynb.
  
  Chạy lần lượt các cell theo thứ tự từ trên xuống.
  
  Đảm bảo thư mục data/ chứa:
  
    sent_train.csv
    
    sent_valid.csv
    
    sentiments.csv

Chạy bằng Command Line
  
  Chạy mô hình baseline và improved model:
    
    python src/test/lab5_test.py

  Chạy mô hình Spark:
    
    python src/test/lab5_spark_sentiment_analysis.py

3. Result Analysis
   
3.1 Baseline Model: Logistic Regression (Task 1 & 2)

Kết quả đánh giá trên tập valid:

Metric	Score
Accuracy	0.50
Precision	0.50
Recall	1.00
F1-score	0.67

Nhận xét:

Mô hình dự đoán rất lệch về một phía (dự đoán hầu hết là 1), dẫn đến Precision thấp dù Recall cao.
→ Hiệu suất thấp, baseline chưa hiệu quả.


3.2 Spark Logistic Regression Pipeline (Task 3)

Accuracy	0.7394

Nhận xét:

Sử dụng Spark và pipeline xử lý văn bản giúp mô hình học đặc trưng tốt hơn → hiệu suất cải thiện rõ rệt từ 0.50 lên ~0.74.


3.3 Improved Model: Naive Bayes (Task 4)

Accuracy	0.7066
(Weighted) F1-score	0.65

Classification Report:

Class	Precision	Recall	F1-score	Support
0	0.88	0.23	0.37	427
1	0.69	0.98	0.81	732

Nhận xét:

Naive Bayes dự đoán tốt lớp 1 nhưng dự đoán rất yếu lớp 0.
  
  Do tập dữ liệu bị mất cân bằng (lớp 1 nhiều hơn lớp 0), NB nghiêng về lớp chiếm đa số.
  
  → Mặc dù cải tiến feature (n-grams) giúp mô hình tốt hơn baseline, nhưng vẫn kém hơn Spark Logistic Regression.

4. Challenges and Solutions
   
Dữ liệu chứa nhiều ký tự đặc biệt và câu không rõ nghĩa   --> 	Áp dụng preprocessing mạnh hơn (regex cleaning, stopword removal).

Mô hình Logistic Regression ban đầu bị overfitting nhẹ	  -->   Điều chỉnh tham số C và bổ sung regularization.

Thời gian training lâu với TF-IDF bigram	                -->   Giảm số feature bằng cách đặt max_features hoặc dùng chi-square selection.
