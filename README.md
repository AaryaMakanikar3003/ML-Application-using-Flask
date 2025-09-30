# 🌸 Iris Dataset ML Project

This project demonstrates how to build and deploy a Machine Learning model using the **Iris dataset**, with API endpoints for easy integration.

## 📂 Project Structure

```
├── Iris.csv                # Dataset
├── train.ipynb             # Jupyter notebook for training & analysis
├── app.py / main.py        # Flask/FastAPI app exposing ML model as API
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## 📊 Dataset

- **Dataset**: [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
- **Features**:  
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- **Target**: Species (`Setosa`, `Versicolor`, `Virginica`)  

## ⚙️ Workflow

1. **Data Preprocessing**  
   - Load dataset (`pandas`)  
   - Handle features & target encoding  
   - Train-test split  

2. **Model Training**  
   - Trained with algorithms like Logistic Regression  
   - Evaluate performance with accuracy & confusion matrix  

3. **API Creation**  
   - Flask app to serve the trained model  
   - Exposes endpoints like:  

   ```
   POST /predict
   {
     "sepal_length": 5.1,
     "sepal_width": 3.5,
     "petal_length": 1.4,
     "petal_width": 0.2
   }
   ```

   **Response:**
   ```json
   {
     "species": "Iris-setosa"
   }
   ```

## 🚀 How to Run

1. Clone the repo  
   ```bash
   git clone <your-repo-link>
   cd <repo-name>
   ```

2. Create virtual environment (optional but recommended)  
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Run training (if needed)  
   ```bash
   jupyter notebook train.ipynb
   ```

5. Start API  
   ```bash
   python app.py
   ```

6. Test API (using curl or Postman)  
   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
   -H "Content-Type: application/json" \
   -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
   ```

## 📈 Results

- Achieved **~97% accuracy** (depending on model used)  
- Model correctly classifies all three Iris species  

## 🛠️ Tech Stack

- **Python** (pandas, numpy, scikit-learn, matplotlib/seaborn)  
- **Flask** for API  
- **Jupyter Notebook** for EDA & training  

