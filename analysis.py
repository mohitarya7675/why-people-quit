import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. AUTOMATICALLY CREATE FOLDERS AND THE MISSING DATA
print("Checking folders and generating dataset...")
os.makedirs('data', exist_ok=True)
os.makedirs('visuals', exist_ok=True)

# Generate 1,000 realistic records mock data
np.random.seed(42)
n_samples = 1000
tenure = np.random.randint(1, 24, n_samples)
contract = np.random.choice(['Monthly', 'Annual'], n_samples, p=[0.6, 0.4])
support_tickets = np.random.poisson(lam=1.5, size=n_samples)
monthly_charges = np.random.uniform(29.99, 119.99, n_samples)
clv = (monthly_charges * tenure * np.random.uniform(0.8, 1.2, n_samples)).round(2)
churn_prob = 0.1 + (contract == 'Monthly') * 0.3 + (support_tickets > 2) * 0.4
churn = np.random.binomial(1, np.clip(churn_prob, 0, 1))

df = pd.DataFrame({
    'CustomerID': range(10001, 10001 + n_samples),
    'Tenure_Months': tenure,
    'ContractType': contract,
    'SupportTickets': support_tickets,
    'MonthlyCharges': monthly_charges.round(2),
    'CLV': clv,
    'Churn': churn
})

# Save the dataset so it's there for next time
df.to_csv('data/customer_churn_data.csv', index=False)
print("✅ Data successfully created and saved to 'data/customer_churn_data.csv'")

# 2. RUN DATA ANALYSIS AND SAVE CHARTS
print("Creating visualization charts...")

# Visual 1: Churn by Contract Type
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='ContractType', hue='Churn', palette='Set2')
plt.title('Why People Quit: Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.legend(title='Quit?', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('visuals/churn_distribution.png', dpi=150)
plt.close()

# Visual 2: CLV vs Tenure
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x='Tenure_Months', y='CLV', hue='Churn', alpha=0.7, palette='coolwarm')
plt.title('When People Quit: Revenue Value vs Staying Time')
plt.xlabel('Months Stayed')
plt.ylabel('Customer Lifetime Value ($)')
plt.tight_layout()
plt.savefig('visuals/clv_vs_tenure.png', dpi=150)
plt.close()

print("✅ Visual charts saved successfully to 'visuals/' folder!")
print("🎉 All done! Your project data and charts are ready.")
