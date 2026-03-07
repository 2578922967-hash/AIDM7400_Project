# 🎬 AIDM7400 Group Project: TMDB Movie Analysis

**Topic:** Decrypting the Code of Box Office Success: A 20-Year Analysis of TMDB Movies  
**题目:** 票房成功的解密代码：TMDB电影数据的20年深度分析

**Team:** 
- Tech (技术组): LI, XU
- Story (文案组): CAI, LIN

## 🎯 Research Questions (RQs) / 核心研究问题
1. **Trend & Geopolitics (趋势与地缘政治):** How have movie genres and production powerhouses shifted over the last 20 years?  
   **流变与格局:** 过去20年间，电影类型的流行趋势和制片国家的版图发生了怎样的地缘与时间转移？

2. **Content & Sentiment (内容与情感):** Do specific narrative elements (based on Overview NLP) correlate with higher Ratings or Revenue?  
   **成功密码:** “叫好”与“叫座”是否矛盾？基于简介的NLP分析，哪类叙事元素的电影更容易获得高票房或高评分？

3. **Prediction (预测模型):** Can we predict a movie's ROI based on its Budget, Cast, and Release Date?  
   **投资回报预测:** 结合预算、档期和卡司阵容，我们能否建立模型预测一部电影的ROI？

## 📂 Project Structure / 项目结构
- `data/`: Raw and processed data (not valid in repo, download from Kaggle).  
  数据文件夹：原始数据与清洗后的数据（不包含在Git仓库中，需从Kaggle下载）。
- `notebooks/`: Analysis Workflow. 分析流程笔记本。
    - `01_Data_Cleaning.ipynb`: JSON parsing (genres, production_companies) & missing value handling.  
      数据清洗：解析JSON字段（类型、制作公司）及缺失值处理。
    - `02_EDA_Descriptive.ipynb`: Basic distribution of Revenue, Budget, and Runtime.  
      探索性数据分析：票房、预算、时长的基础分布。
    - `03_Analysis_RQ1_Trends.ipynb`: Visualization of Genre trends over time.  
      RQ1分析：电影类型随时间的流行趋势可视化。
    - `04_Analysis_RQ2_NLP.ipynb`: Keyword extraction & Sentiment analysis on Movie Overviews.  
      RQ2分析：基于电影简介的关键词提取与情感分析。
    - `05_Modeling_RQ3_ROI.ipynb`: Machine Learning models for ROI prediction.  
      RQ3建模：用于预测ROI的机器学习模型。
- `reports/`: Generated figures and draft texts.  
  报告文件夹：生成的图表和文案草稿。

## 📅 Project Roadmap & Status (项目进度表)

### Phase 1: Data Foundation (基础数据建设)
- [x] **Data Collection**: Download TMDB dataset (Status: Done)
- [x] **Data Cleaning**: Handle missing values, JSON parsing, deduplication (Status: Done)
- [x] **Exploratory Data Analysis (EDA)**: General distribution, correlations (Status: Done)

### Phase 2: In-depth Analysis (深度分析 - Tech Team)
#### RQ1: Trend & Geopolitics (趋势演变)
- [ ] **Genre Evolution**: Analyze genre popularity shifts over 20 years
- [ ] **Production Power**: Visualize the rise of production countries
- [ ] **Visualization Output**: Dynamic charts/maps for report

#### RQ2: Content & Success Factors (内容密码)
- [ ] **NLP Preprocessing**: Tokenize and clean movie overviews
- [ ] **Keyword Extraction**: Identify high-revenue keywords
- [ ] **Correlation Analysis**: Narrative elements vs. Box office/Ratings

#### RQ3: Prediction Modeling (预测模型)
- [ ] **Feature Engineering**: Process cast, director, and seasonality features
- [ ] **Model Training**: Build ROI prediction models (Regression/Classification)
- [ ] **Model Evaluation**: Assess accuracy and feature importance

### Phase 3: Reporting & Presentation (报告与展示 - Story Team)
- [ ] **Drafting**: Combine data insights into a cohesive narrative
- [ ] **Visualization Polish**: Finalize charts for aesthetic appeal
- [ ] **Video Script**: Write script based on analysis findings
- [ ] **Final Video**: Record and edit the presentation
- [ ] **Submission**: Final review and submit

### 完整项目结构
- `data/raw/`：存放原始数据 (TMDB csv)。
- `data/processed/`：存放清洗后的数据。
- `notebooks/`：存放分析用的 Jupyter Notebooks。
- `reports/figures/`：存放导出的图表。
- `reports/drafts/`：存放文案草稿。
- `references/`：存放参考文献。
- `src/`：存放 Python 源码。
- `video/scripts/ & video/slides/`：存放视频脚本和 PPT

## 🛠 Setup / 启动指南
1. Download `TMDB_movie_dataset_v11.csv` from Kaggle.  
   从Kaggle下载原始数据 `TMDB_movie_dataset_v11.csv`。
2. Place it in `data/raw/`.  
   将其放入 `data/raw/` 目录。
3. Run `pip install -r requirements.txt`.  
   运行命令安装依赖。
4. Run `notebooks/01_Data_Cleaning.ipynb` first to generate processed data.  
   首先运行 `01_Data_Cleaning.ipynb` 以生成清洗后的数据。
