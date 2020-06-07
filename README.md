# Predicting-stock-price-by-social-media
Use AI and data from social media to predict the TAIEX(Taiwan stock exchange weighted index)

團隊使用2016-2018三年的財經新聞及論壇討論資料，從中取出關鍵字列表，建構向量空間，預測大盤指數的漲跌。在團隊中負責資料的前處理（包括資料清洗、關鍵字擷取及建構向量空間）和random forest的模型建構。
使用監督式學習之分類演算法(Naive Bayes、KNN、Decision Tree、Random Forest、SVM、Gradient Boosting)，評估模型準確率時得到Precision rate: 95%，recall rate: 75%。

Note: 由於資料檔案過大，放置於 https://drive.google.com/drive/folders/1qScXQL-vwJH7FLw9EdvHks-2esNkbqaW?usp=sharing


----------------

### 詳細建構關鍵字空間向量方式

我們選使用台灣加權股價指數(大盤指數y9999)作為預測對象，在關鍵字排序的部分使用TF-IDF作為指標，同時踢除出現篇數(DF)低於總篇數0.1%的字詞、出現次數(TF)及出現篇數(DF)的上限同為第99.9百分位數，加入論壇討論量作為篩選標準；漲跌的標準則以20日移動平均為基準判斷，最後預測5日後的指數漲跌。

### 實際步驟
關鍵字取得的部分：
1. 將文章切成2~6gram
2. 移除停用詞及DF小於篇數0.1%的詞，
3. 計算TF-IDF
4. 移除被長詞包含，且DF相差20%以內的字詞  

接著，依據股價的波動，將文章分成上漲與下跌兩類，分類標準為：
1. 以20日加權移動平均線為基準
2. 股價反應日數 N = 5
3. 論壇討論量超過第75百分位數
