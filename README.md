# Predicting-stock-price-by-social-media
Use AI and data from social media to predict the TAIEX(Taiwan stock exchange weighted index)

團隊使用2016-2018三年的財經新聞及論壇討論資料，從中取出關鍵字列表，建構向量空間，預測大盤指數的漲跌。在團隊中負責資料的前處理（包括資料清洗、關鍵字擷取及建構向量空間）和random forest的模型建構。
使用監督式學習之分類演算法(Naive Bayes、KNN、Decision Tree、Random Forest、SVM、Gradient Boosting)，評估模型準確率時得到Precision rate: 95%，recall rate: 75%。

Note: 由於資料檔案過大，放置於 https://drive.google.com/drive/folders/1qScXQL-vwJH7FLw9EdvHks-2esNkbqaW?usp=sharing


----------------

### 詳細建構關鍵字空間向量方式

我們選擇使用台灣加權股價指數(大盤指數y9999)作為預測對象，在關鍵字排序的部分使用TF-IDF作為指標，同時踢除出現篇數(DF)低於總篇數0.1%的字詞、出現次數(TF)及出現篇數(DF)的上限同為第99.9百分位數，加入論壇討論量作為篩選標準；漲跌的標準則以20日移動平均為基準判斷，最後預測5日後的指數漲跌。

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

再將文章和取出的200個關鍵字(上漲下跌各100)製成count matrix

### 模型選用
測試 Naive Bayes、KNN、Decision Tree、Random Forest、SVM、Gradient Boosting 六種模型  
- 將資料標準化，並測試用PCA簡化資料維度（成效不顯著）
- 以GridSearchCV進行4-fold cross validation優化參數
- 設定class weights處理不平衡的訓練數據
- 以Voting Classifier讓模型投票  

最後選擇準確率最佳的SVM與Random Forest
(Precision = 95.06~97.83%, Recall = 74.06~74.54%)

### 指數漲跌預測

- 用前述關鍵字作為基準，計算每個關鍵字在每一天新聞中的出現次數
- 若五日後的當日上漲價格超過0.3%，則該日視為上漲;下跌超過0.3%則視為下跌
- 訓練資料中的新聞篩選標準與前述相同
- 每次以三個月的新聞作為訓練資料，並將後一月每一天的新聞作為個別的測試資料
- 模型選用考量SVM和Random Forest的 Voting Classifier
- 若預測上漲的新聞數為預測下跌的1.2倍， 則預測股價上漲，反之亦然;若介於之間，則預測為不出手

結果得到Precision = 59.09%，Recall = 40.21%

### 未來可改善部分

- 使用Monpa或CkipTagger斷詞
- 模型的參數測試:畫出ROC curve權衡不同閥值參數，優化(FPR, TPR)
