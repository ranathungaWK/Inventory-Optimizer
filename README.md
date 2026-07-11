# 🛒 Grocery Inventory Optimization Platform

## 1. What is the Project?
An end-to-end data pipeline and decision-support system modeled after supermarket supply chains (e.g., Corporación Favorita). It ingests large-scale daily transaction records, catalog details, calendar events, and economic markers to automatically calculate daily stock recommendations.

---

## 2. What is the Problem Solved?
Traditional forecasting only predicts raw demand, ignoring the financial trade-off between:
*   **Understocking (Stockouts):** Lost revenue, customer dissatisfaction, and brand switching when demand is higher than stock.
*   **Overstocking (Spoilage/Wastage):** Capital tied up in excess inventory and direct financial loss from expired perishable items.

A model optimized purely for forecast error (e.g. MAE or RMSE) fails to balance these opposing costs. This platform explicitly optimizes the financial impact rather than statistical error alone.

---

## 3. How We Solve It (Methodology & Architecture)
We process raw grocery sales records through a **Medallion Data Lakehouse** and apply the **Newsvendor Model** to find the optimal order point.

### A. Medallion Lakehouse Flow
1.  **Bronze Ingestion ([01_ingest_bronze.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/bronze/01_ingest_bronze.ipynb)):** Memory-safe, chunked streaming of raw CSV files (including the 4.65 GB transactions log) into highly compressed Snappy Parquet format.
2.  **Silver Standardization ([silver/](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/silver/)):**
    *   [01_build_sales_fact.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/silver/01_build_sales_fact.ipynb): Structures the main transaction fact table (~125.5M rows).
    *   [02_build_item_dimension.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/silver/02_build_item_dimension.ipynb): Categorizes item metadata and maps bilingual details.
    *   [03_build_store_dimension.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/silver/03_build_store_dimension.ipynb): Cleans physical store layout attributes.
    *   [04_build_calendar_dimension.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/silver/04_build_calendar_dimension.ipynb): Aligns dates, engineers holiday/weekend flags, and translates descriptions utilizing [translations.py](file:///d:/Demand_forecasting/inventory-lakehouse/src/config/translations.py).
    *   [05_build_oil_dimension.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/silver/05_build_oil_dimension.ipynb): Integrates economic oil price indices, interpolates gaps, and computes rolling averages.
3.  **Gold Aggregation ([01_analysing_data.ipynb](file:///d:/Demand_forecasting/inventory-lakehouse/notebooks/gold/01_analysing_data.ipynb)):** Prepares aggregated fact-dimension joins and visualizations for modeling.

### B. Operations Research Optimization
The platform utilizes the **Newsvendor Formula** to determine the optimal service level ($SL$):

$$SL = \frac{C_u}{C_u + C_o}$$

Where:
*   $C_u$ = Understock cost (margin margin + penalty)
*   $C_o$ = Overstock cost (item cost - salvage value)

The final order quantity is set to satisfy the demand up to the calculated $SL$ percentile of the predicted distribution ($Q = F^{-1}(SL)$).

---

## 4. Technologies Used
*   **Data Lakehouse Format:** Apache Parquet (Snappy Compressed)
*   **Data Processing:** PyArrow (memory-safe streaming & parsing), Pandas, NumPy
*   **Analytics & Visuals:** Jupyter Notebooks, Matplotlib, Seaborn
*   **Operating System/Infrastructure:** Python 3.9+, psutil (resource tracking)
*   **MLOps Architecture (Planned):** FastAPI (Prediction API), Streamlit (Dashboard), MLflow (Experiment tracking), Evidently AI (Drift monitoring)

---

## 5. Expected Outcomes
*   **Optimal Order Quantities:** Automatic calculation of recommended SKU replenishment values per store.
*   **Minimized Operational Costs:** Reduced inventory wastage for perishables and eliminated missed sales from stockouts.
*   **Efficient Big Data Handling:** Ingestion and compression of a 4.65 GB raw dataset down to **~350 MB** (a 13.2x reduction), processing over 125 million rows in seconds.
*   **Bilingual Cleansing:** Standardized, clean English-Spanish product catalogs and holiday schemas for analysis.
