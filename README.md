# Mental_Health_Prediction_System
---

## **📊 Dataset & Preprocessing**
### **📌 Dataset Sources**
- **Mental Health in Tech Survey**
- **Depression and Anxiety Symptoms Dataset**
- **WHO Mental Health Database**

### **📌 Preprocessing Steps**
1️⃣ **Missing Value Handling** – Fill missing values with appropriate strategies.  
2️⃣ **Feature Encoding** – Convert categorical variables (e.g., Gender) to numerical format.  
3️⃣ **Normalization** – Scale numerical features for model training.  
4️⃣ **Feature Selection** – Select the most impactful features for prediction.  

**Selected Features:**
- `Age`
- `Gender`
- `Family History`
- `Benefits`
- `Care Options`
- `Anonymity`
- `Leave`
- `Work Interfere`

---

## **📌 Model Selection**
We compared multiple models:
| Model                  | Accuracy |
|------------------------|---------|
| Logistic Regression    | **78.57%** ✅ |
| Random Forest         | 73.42% |
| Deep Neural Network   | 52.38% |
| BERT Transformer      | (Optional, not used in final version) |

**Final Choice:**  
✅ **Logistic Regression (Best Accuracy & Speed)**  
✅ **Random Forest (For Interpretability)**  
✅ **DNN (For Deep Learning-based Analysis)**  

---

## **📌 How to Run the Inference Script**
### **1️⃣ Install Dependencies**
Ensure you have **Python 3.8+** installed. Then, install required libraries:

```bash
pip install -r requirements.txt
