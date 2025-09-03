# GTC ML Project 1: Hotel Bookings Data Preprocessing Pipeline

## 📋 Project Overview

This project focuses on building a robust data preprocessing pipeline for hotel booking cancellation prediction. The goal is to transform raw hotel booking data from a Property Management System (PMS) into a clean, machine-learning-ready dataset.

## 🎯 Business Problem

The revenue team has identified that last-minute booking cancellations significantly impact profitability. This project prepares the raw data for a future machine learning model that will predict booking cancellations, ensuring the model's success through high-quality data preprocessing.

## 📊 Dataset Description

**Source**: `hotel_bookings-hotel_bookings.csv`  
**Original Size**: ~119,390 rows × 32 columns  
**Final Size**: 87,377 rows × 22 columns (after preprocessing)

### Key Features:
- **Target Variable**: `is_canceled` (0 = not canceled, 1 = canceled)
- **Booking Information**: lead_time, arrival_date, stays_in_weekend_nights, stays_in_week_nights
- **Guest Information**: adults, children, babies, country
- **Booking Details**: meal, market_segment, distribution_channel, deposit_type, customer_type
- **Pricing**: adr (average daily rate)
- **Operational**: agent, company, required_car_parking_spaces, total_of_special_requests

## 🔄 Preprocessing Pipeline

### Phase 1: Exploratory Data Analysis (EDA) & Data Quality Report
- **Data Loading**: Import and initial inspection of the dataset
- **Summary Statistics**: Generate descriptive statistics and data info
- **Missing Values Analysis**:
  - Company: 94% missing
  - Agent: Moderate missing values
  - Country: Low missing values
  - Children: Very few missing values
- **Outlier Detection**: Identified significant outliers in `adr` (3,793) and `lead_time` (3,005) using IQR method
- **Data Quality Issues Identified**:
  - High missing values in company column
  - Outliers in pricing and timing features
  - Duplicate records
  - Incorrect data types for date columns

### Phase 2: Data Cleaning (Core Phase)
- **Missing Values Handling**:
  - `company`: Filled with 'None' (categorical placeholder)
  - `agent`: Filled with 0 (numerical placeholder)
  - `country`: Filled with mode (most frequent country)
  - `children`: Filled with median value
- **Duplicate Removal**: Removed 32,013 exact duplicate rows
- **Outlier Treatment**: Capped `adr` values above 1000 to prevent model skewing
- **Data Type Corrections**: Converted date columns to proper datetime format

### Phase 3: Feature Engineering & Preprocessing
- **New Features Created**:
  - `total_guests = adults + children + babies`
  - `total_nights = stays_in_weekend_nights + stays_in_week_nights`
  - `is_family` = Binary flag (1 if children/babies > 0, else 0)
- **Categorical Encoding**: Applied one-hot encoding to categorical variables
- **Data Leakage Prevention**: Removed `reservation_status` and `reservation_status_date` columns (contain future information)
- **Train-Test Split**: 80/20 split with random_state=42 for reproducibility

## 📁 File Structure

```
gtc-ml-project1-hotel-bookings/
├── Project_01.ipynb              # Main preprocessing notebook
├── hotel_bookings-hotel_bookings.csv  # Raw dataset
├── GTC ML Project 1.pdf         # Project requirements
└── README.md                    # This file
```



## 📈 Key Results

- **Data Quality Improvements**:
  - Eliminated all missing values
  - Removed 26.8% duplicate records
  - Capped extreme outliers in pricing data
  - Fixed data type inconsistencies

- **Feature Engineering**:
  - Added 3 new derived features
  - Encoded all categorical variables
  - Prevented data leakage
  - Prepared data for ML algorithms

## 🛠️ Technical Details

- **Libraries Used**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Encoding Method**: One-hot encoding for categorical variables
- **Outlier Method**: IQR-based capping (upper limit = 1000 for adr)
- **Split Strategy**: Stratified 80/20 train-test split
- **Data Types**: All features converted to numerical for ML compatibility

## 📝 Notes

- The preprocessing pipeline is designed to be reproducible and production-ready
- All transformations are justified and documented in the notebook
- The final dataset is ready for machine learning model training
- Data leakage has been carefully prevented to ensure real-world model validity
