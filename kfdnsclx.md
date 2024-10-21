# **Scam Conversation Classification Pipeline**


### **1. Data Gathering**
   - **Description**: This step involves collecting real-world conversations that may involve scams. The data can be sourced from publicly available datasets (e.g., SMS Spam Collection, social media conversations), user-reported scams, or from internal systems.
   - **Why it’s needed**: High-quality and representative data is the foundation of model performance. For robust classification, the data should capture various scam tactics.
   - **Relevant Links**:
     - [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
     - [Raju and 40 theives](https://www.rbi.org.in/commonperson/English/Scripts/BasicBankingNew.aspx)

---

### **2. Data Annotation & Labeling**
   - **Description**: The gathered data must be annotated for labeling conversations as “scam” or “not scam.” Labeling can be done manually, using rule-based systems, or by leveraging pre-trained models.
   - **Why it’s needed**: Supervised learning algorithms require labeled data for training and evaluation.

---

### **3. Synthetic Data Creation**
   - **Description**: To augment the dataset, synthetic data generation is used. You can use language models (e.g., ChatGpt, LLama) to generate synthetic scam and non-scam conversations based on patterns in the real dataset.
   - **Why it’s needed**: Synthetic data balances the dataset if one class (e.g., scam) is under-represented. It helps in improving model generalization by exposing it to varied scenarios.
   - **Relevant Links**:
     - [ChatGpt](https://chatgpt.com)
     - [Using GPT-3 for Synthetic Data](https://arxiv.org/abs/2109.05537)

---

### **4. Data Preprocessing**
   - **Lowercasing**: Standardize the text to lowercase to reduce variance in word representations.
   - **Tokenization**: Break down sentences into words/tokens.
   - **Why it’s needed**: Preprocessing ensures consistency in data and reduces noise, allowing models to focus on more meaningful patterns in the text.
   - **Relevant Links**:
     - [Text Preprocessing in NLP](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)

---

### **5. Named Entity Recognition (NER)**
   - **Description**: Use a pre-trained model (e.g., BERT-CRF) to extract key entities from conversations, such as phone numbers, URLs, and names. NER helps in identifying known scam patterns.
   - **Why it’s needed**: Scammers often use named entities (e.g., fake company names, links). Identifying these entities allows the model to flag them as suspicious.
   - **Relevant Links**:
     - [NER using Transformers](https://huggingface.co/docs/transformers/task_summary#token-classification)

---

### **6. Feature Engineering**
   - **Description**: Beyond the raw text, features like message length, time between messages, or the number of detected entities can be added. (Future pipeline)
   - **Why it’s needed**: These additional features can provide useful signals that might not be captured by text alone.
   - **Relevant Links**:
     - [Feature Engineering in NLP](https://www.analyticsvidhya.com/blog/2021/04/a-guide-to-feature-engineering-in-nlp/)

---

### **7. Model Selection Criteria**
   - **Model Type**: Choose a model architecture (e.g., BERT, Electra, DistilBERT) based on factors like performance, model size, and inference speed.
   - **Why it’s needed**: Depending on constraints like inference time, model accuracy, and available compute, you need to select the most appropriate model.
   - **Relevant Links**:
     - [BERT](https://huggingface.co/blog/bert-101)

---

### **8. Data Splitting (Train, Validation, Test)**
   - **Description**: Split the dataset into training, validation, and test sets (e.g., 70/20/10 split) to ensure proper model evaluation.
   - **Why it’s needed**: This ensures that the model doesn’t overfit and generalizes well to unseen data.

---

### **9. Fine-Tuning**
   - **Description**: Use a pre-trained transformer model (e.g., Google Electra) and fine-tune it on your scam classification task using your prepared dataset.
   - **Why it’s needed**: Fine-tuning allows the model to adapt from general language understanding to the specific task of scam detection.
   - **Relevant Links**:
     - [Fine-Tuning Transformers](https://huggingface.co/transformers/training.html)

---

### **10. Model Evaluation**
   - **Metrics**: Evaluate the fine-tuned model using metrics like accuracy, F1 score, precision, and recall on the validation and test sets.
   - **Why it’s needed**: These metrics indicate the model’s performance on the classification task, ensuring it can identify scam messages effectively.
   - **Relevant Links**:
     - [Model Evaluation Metrics](https://towardsdatascience.com/metrics-for-evaluating-machine-learning-classification-models-python-example-59b905e079a5

---

### **11. Conversion to ONNX Format**
   - **Description**: Convert the trained and optimized model to ONNX (Open Neural Network Exchange) format for better cross-platform compatibility and inference efficiency.
   - **Why it’s needed**: ONNX models allow for hardware acceleration and faster inference across various devices.
   - **Relevant Links**:
     - [ONNX Tutorial](https://onnx.ai/tutorials/)

---
### **12. Model Optimization (Pruning, Quantization)**
   - **Description**: To deploy on-device, optimize the model by pruning unnecessary layers and quantizing it to reduce model size and computation time. Quantization involves converting model weights from float32 to int8, reducing the model’s size and computational requirements.
   - **Why it’s needed**: For efficient deployment on edge devices (e.g., mobile phones), the model must be lightweight without sacrificing too much accuracy.
   - **Relevant Links**:
     - [Model Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)


---

### **13. Deployment on Device**
   - **Description**: Load the quantized ONNX model onto mobile or IoT devices for real-time scam conversation classification.
   - **Why it’s needed**: Running the model on-device enables privacy-preserving and low-latency inference, essential for real-time scam detection.
   - **Relevant Links**:
     - [Running ONNX Models on Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)

---

### **14. Monitoring and Model Updating**
   - **Description**: After deployment, continue monitoring the model’s performance. Use tools to capture edge cases where the model fails, and periodically retrain the model with new data.
   - **Why it’s needed**: Regular updates ensure the model adapts to new scam tactics and evolving language patterns.

---

### **Diagram Overview**

```plaintext
+-----------------+
| 1. Data         |
|    Gathering    |
+-----------------+
        |
        v
+-----------------+
| 2. Annotation & |
|    Labeling     |
+-----------------+
        |
        v                                                                
+-----------------+
| 3. Synthetic    |
|    Data         |
|    Labeling     |
+-----------------+
        |
        v                                                                
+-----------------+
| 4. Data         |
|    Preprocessing|
+-----------------+
        |
        v
+-----------------+
| 5. Named Entity |
|    Recognition  |
+-----------------+
        |
        v
+-----------------+
| 6. Feature      |
|    Engineering  |
+-----------------+
        |
        v
+-----------------+
| 7. Model        |
|    Selection    |
+-----------------+
        |
        v
+-----------------+
| 8. Data Splitting|
+-----------------+
        |
        v
+-----------------+
| 9. Fine-Tuning   |
+-----------------+
        |
        v
+-----------------+
|10. Model Eval    |
+-----------------+
        |
        v
+-----------------+
|11. Optimization  |
+-----------------+
        |
        v
+-----------------+
|12. Conversion to |
|    ONNX Format   |
+-----------------+
        |
        v
+-----------------+
|13. On-Device     |
|    Deployment    |
+-----------------+
        |
        v
+-----------------+
|14. Monitoring &  |
|    Updating      |
+-----------------+
```