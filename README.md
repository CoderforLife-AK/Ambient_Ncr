# Ambient-NCR 🍃
> **AI-Powered Regional Air Quality Tracker & Forecasting Assistant**

Ambient-NCR is an intelligent environmental monitoring web application designed to track, analyze, and forecast regional Air Quality Index (AQI) levels across the National Capital Region. 

Built as a core project milestone for the **1M1B Sustainability Internship Framework**, this application aligns directly with **UN Sustainable Development Goal (SDG) Target 11.6**: *To reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality.*

---

## 🔗 Live Application Link
👉 **https://ambientncr-mlproject.streamlit.app/

---

## 🎯 Project Purpose & Impact
Urban centers face critical public health challenges due to shifting ambient air metrics. Ambient-NCR bridge this gap by converting complex climate data points into live, actionable health advisories. 
* **The Problem:** Raw pollutant concentrations ($PM_{2.5}$, $PM_{10}$, $NO_2$) are difficult for everyday citizens to translate into immediate health precautions.
* **The Solution:** A high-speed Machine Learning pipeline that analyzes real-time environmental metrics and instantly provides color-coded risk alerts and protective guidelines.

---

## 🛠️ Tech Stack & Architecture

The system uses a lightweight, optimized micro-architecture for instant cloud execution:
* **Machine Learning Framework:** Scikit-Learn (Random Forest Regressor)
* **Data Processing & Analytics:** Pandas, NumPy
* **Frontend Web Dashboard:** Streamlit UI Engine
* **UI Styling:** Custom CSS & Tailwind-inspired Dark Slate Configuration

---

## 🧠 How the Machine Learning Core Works

Instead of relying on slow external server connections, Ambient-NCR processes data natively in the cloud container:[delhi_pollution.zip] ──> Extracts CSV ──> One-Hot Encodes Regions ──> Trains Random Forest ──> Active Model Core
1. **Data Ingestion:** When the application boots, it automatically extracts and unzips a 5 MB localized dataset repository (`delhi_pollution.zip`) directly in system memory.
2. **Preprocessing:** It automatically cleans column configurations, handles missing null thresholds, and dynamically sets up regional tracking labels via **One-Hot Encoding**.
3. **Optimized Training:** It initializes a high-performance `RandomForestRegressor` ensemble model. To ensure the cloud link loads instantly, the ensemble structure is highly optimized to map splits seamlessly in 1–2 seconds.
