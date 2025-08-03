# Montana Elk Harvest Clustering

This project analyzes elk harvest estimates in Montana using **K-Means Clustering** to uncover patterns and natural groupings across different hunting units.

## 📊 Project Overview
The dataset contains **272 samples** and **11 features**, including:
- Hunting District
- Residency
- Total Harvest
- Bulls
- Cows
- Different hunting methods

Since the dataset had **no labels**, unsupervised learning was used to find patterns in the data.  

### 🔑 Key Steps:
1. **Data Cleaning & Preprocessing**  
   - Converted `Residency` to integers  
   - Removed unnecessary columns  
   - Prepared the dataset for clustering  

2. **Dimensionality Reduction with PCA**  
   - Used **Principal Component Analysis (PCA)** to reduce features for visualization  
   - The first **two principal components explained 97.74% of the variance**, allowing for clear 2D plotting  

3. **K-Means Clustering**  
   - Applied K-Means to group hunting units based on harvest data  
   - Chose **4 clusters**, which produced distinct groupings  

4. **Insights**  
   - Wrote a script to trace cluster points back to the original dataset  
   - Found that clusters were **mostly influenced by geographic location**  
   - The results suggest that **elk behavior and hunting strategies vary by terrain and region**

## 📈 Results
- The clustering revealed **4 distinct groups** when plotted on the PCA-reduced dimensions.  
- The analysis provided insights into how hunting units differ across Montana.  
- More analysis could be done to explore **why** certain areas group together (e.g., terrain, hunting regulations, or herd size).

## 🛠 Tools & Libraries
- Python  
- Pandas  
- NumPy  
- Matplotlib
