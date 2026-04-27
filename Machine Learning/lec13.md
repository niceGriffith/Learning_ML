KNN: K-NEAREST NEIGHBOURSCS456-MACHINE LEARNING SPRING 2023Rahul Vishwakarma, Jyothish Kumar JSchool of Computer Sciences,National Institute of Science Education and Research, Bhubaneshwar,Homi Bhabha National InstituteFebruary 19, 2023PART I: THEORY1 Introduction .......................................................................... 41.1 General Information ........................................................... 41.2 Special Points .................................................................... 62 Applications ........................................................................... 73 Algorithm .............................................................................. 83.1 Overview and Psuedocode ................................................ 83.2 Distance Metrics ................................................................ 9PART II: DEMONSTRATION1 Data collection and processing ............................................. 112 Code .................................................................................... 122.1 Image to Vector ................................................................ 122.2 General classification scheme .......................................... 132.3 Handwriting recognition .................................................. 143 Observations and Results .................................................... 16Part I: THEORY1. IntroductionGeneral InformationFigure. Visual representation of k-NN classification [Gandhi n.d.](Visual representation displays Class A, Class B, Class C, and a new point $P_{t}$ with arrows pointing to nearest neighbors)k-NN or K-Nearest Neighbour is a supervised classification algorithm. When a new piece of data is received, it's compared against all existing pieces of data for similarity. Once the top $k^{\prime}$ nearest neighbors are identified, a majority vote is taken from these k data-points and the new point is assigned the majority vote as its class.Here 'k' is the hyper-parameter responsible for controlling the inductive bias. [Harrington 2012]There is no training involved in this algorithm. These kinds of models are called Instance-based learning.While chosen k can be both odd/even, Odd values of k is preferred since majority voting is done to get the classifying radius. If voting is near 50:50, weightage can be given to points to prevent anomalies.For large values of k, this model becomes computationally expensive.Computational geometry concepts such as Voronoi diagrams are used for finding the neighborhood.Figure. Voronoi Diagram [Bellelli n.d.]Special PointsThe algorithm has to carry around the full dataset; for large datasets, this implies a large amount of storage. In addition, you need to calculate the distance measurement for every piece of data in the database, and this can be cumbersome.An additional drawback is that KNN doesn't give you any idea of the underlying structure of the data; you have no idea what an "average" or "exemplar" instance from each class looks like.Pros: High accuracy, insensitive to outliers, no assumptions about data.Cons: Computationally expensive, requires a lot of memory.Works with: Numeric values and nominal values.2. ApplicationsThe following example problem statements can be well addressed using KNN as a classifier.Handwriting recognition: Given enough samples of handwritten specimen, a kNN classifier can be used to identify any new letter/number based on it's appearance similarity with the sample data. We will explore the implementation of this example in Part 2. [Code and Dataset obtained from [GitHub - pbharrin/machinelearninginaction: Source Code for the book: Machine Learning in Action published by Manning - github.com n.d.]]Match making on dating sites: Classifier can match like-minded people using their inputs collected at the time of registration.Movie classification: Classification of any given movie into genres such as romance, action, comedy etc. based on various features.etc.3. AlgorithmOverview and PsuedocodeAfter collection and preparation of data. The foundational steps involved in k-NN algorithm are as follows:Distance calculationSorting dictionaryVoting with lowest k distancesPsuedocode is given as:For every point in our dataset:calculate the distance between inX and the current pointsort the distances in increasing ordertake k items with lowest distances to inXfind the majority class among these itemsreturn the majority class as our prediction for the class of inXDistance MetricsFor the algorithm to work best on a particular dataset we need to choose the most appropriate distance metric accordingly. Some of the commonly used distance matrices for KNN are:Euclidean Distance$d=\sqrt{\sum_{i=1}^{n}(x_{i}-y_{i})^{2}}$Minkowski Distance$d=(\sum_{i=1}^{n}|x_{i}-y_{i}|^{p})^{1/p}$Manhattan Distance$d=\sum_{i=1}^{n}|x_{i}-y_{i}|$Part II: DEMONSTRATION1. Data collection and processingAim: To design a classifier that recognizes a given image of a hand-written figure of a number between 0 to 9.Data-set: Over 2000 image samples of hand written numbers (0-9), approximately 200 samples per digit. Data made available in public domain by [Alpaydin and Kaynak n.d.].Obtained images are equivalent to a $32\times32$ matrix of 0s and 1s. Dark or inked areas of image reprented by 1s and blank areas by 0s. [Fig. 3] These matrices are then converted to vectors of $1 \times 1024$. ^1Figure. Binary representation of a sample image.^1 To make demonstration/replication easy, The converted vector has been provided for download [here.]2. CodeImage to VectorThe python code used to convert image files to $1\times1024$ vector on binary is given below:def img2vector (filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
Output:>>> testVector = kNN.img2vector('testDigits/0_13.txt')
>>> testVector [0,0:31]
array ([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
>>> testVector [0,32:63]
array ([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
Figure. Output for the function img2vector().General classification schemeThe below function (classify0()) is used to classify any data-set by calling the function with these 4 parameters:Test vectorTraining matrixList of labelsK value (Hyperparameter)def classify0 (inX, dataSet, labels, k):
    dataSetSize  dataSet.shape[0]
    diffMat tile (inX, (dataSetSize,1))
    sqDiffMat = diffMat**2
    dataSet
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels [sortedDistIndicies[i]]
        classCount [votellabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted (classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount [0] [0]
Handwriting recognitionWe have compiled the procured dataset into Test and Training folders [Download]. Though the Training folder contains 2000 data-points (each being 1024-entry Floating Point Vectors).The python code used for handwriting recognition from our dataset is given in the next slide. The code tests the recognition of 900 Samples given in the Test folder.Function handwritingClassTest(); is a self contained classifier that tests our classifier. The code does three things:Parces each test file using OS functions in python.uses the previously discussed img2vector() function to convert the sample to vector.sends the vector through the classification function (classify0()) along with the training matrix, lables and K value. to obtain the output.evaluates the output for error and reports the results along with accuracy of recognition / classification. The output of this function is shown in Fig. 5.def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len (trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int (fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #load the training set
        #take off .txt
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    
    mTest len (testFileList)
    for i in range (mTest):
        testFileList = listdir('testDigits')
        errorCount = 0.0
        #iterate through the test set
        fileNameStr = testFileList [i]
        fileStr  fileNameStr.split('.')[0]
        classNumStr = int (fileStr.split('_')[0])
        #take off .txt
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0 (vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
        
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
3. Observations and Results>>> kNN.handwritingClassTest()
the classifier came back with: 0, the real answer is: 0
the classifier came back with: 0, the real answer is: 0
the classifier came back with: 7, the real answer is: 7
the classifier came back with: 7, the real answer is: 7
the classifier came back with: 8, the real answer is: 8
the classifier came back with: 8, the real answer is: 8
the classifier came back with: 8, the real answer is: 8
the classifier came back with: 6, the real answer is: 8
the classifier came back with: 7, the real answer is: 9

the total number of errors is: 11
the total error rate is: 0.011628
Figure. Output of function handwritingClassTest()The error rate obtained in our experiment $=1.2\%$For each 900 test cases we had to do 2000 distance calculations on a 1024-entry floating point vector. Though easy to implement, This is resource extensive and inefficient. Additionally our dataset (.txt file) was also 2 mb.REFERENCES IAlpaydin, E and C Kaynak (n.d.). "Optical recognition of handwritten digits data set. UCI Machine Learning Repository (1998)". In: URL https://archive. ics. uci. edu/ml/datasets/Optical+ Recognition+of+ Handwritten+ Digits ().Bellelli, Francesco (n.d.). The fascinating world of Voronoi diagrams towardsdatascience.com. https://towardsdatascience.com/the-fascinating-world-of-voronoi-diagrams-da8fc700fa1b. [Accessed 19-Feb-2023].Gandhi, Sai Kumar (n.d.). Finding out Optimum Neighbours (n) number in the KNN classification using Python medium.com. https://medium.com/analytics-vidhya/finding-out-optimum-neighbours-n-number-in-the-knn-classification-using-python-9bdcfefff58c. [Accessed 19-Feb-2023].GitHub - pbharrin/machinelearninginaction: Source Code for the book: Machine Learning in Action published by Manning - github.com (n.d.). https://github.com/pbharrin/machinelearninginaction.git. [Accessed 19-Feb-2023].Harrington, Peter (Apr. 2012). Machine Learning in Action. en. London, England: Simon and Schuster.