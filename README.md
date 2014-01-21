###SVMLightClassifier

A simple static library wrapping SVMLight meant for classification using HOG features.


####Usage

Project using this static library shall contain the following header:

```C++
#ifndef __SVMLIGHTLIB_H__
#define __SVMLIGHTLIB_H__

#include <vector>
#include <string>
#include <fstream>

namespace SVMLight
{
    class SVMTrainer
    {
    private:
        std::fstream featuresFile_;
        std::string featuresFileName_;
    public:
        SVMTrainer(const std::string& featuresFileName);
        void writeFeatureVectorToFile(const std::vector<float>& featureVector, bool isPositive);
        void trainAndSaveModel(const std::string& modelFileName);
    };

    class SVMClassifier
    {
    public:
        SVMClassifier(const std::string& modelFileName);
        std::vector<float> getDescriptorVector();
    };
}

#endif
```

#####Training

The training can use the `SVMTrainer` wrapper the following way:

```C++
    // we are going to use HOG to obtain feature vectors:
    HOGDescriptor hog;
    hog.winSize = Size(32,48);

    // and feed SVM with them:
    SVMLight::SVMTrainer svm("features.dat");

    size_t posCount = 0, negCount = 0;
    for (size_t i = 1; i <= 800; ++i)
    {
        // in this concrete case I had files 0001.JPG to 0800.JPG in both "positive" and "negative" subfolders:
        std::ostringstream os;
        os << TRAINING_SET_PATH << "positive\\" << std::setw(4) << std::setfill('0') << i << ".JPG";
        Mat img = imread(os.str(),CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data)
            break;
        
        // obtain feature vector:
        vector<float> featureVector;
        hog.compute(img, featureVector, Size(8, 8), Size(0, 0));
        
        // write feature vector to file that will be used for training:
        svm.writeFeatureVectorToFile(featureVector, true);                  // true = positive sample
        posCount++;

        // clean up:
        featureVector.clear();
        img.release();              // we don't need the original image anymore
        os.clear(); os.seekp(0);    // reset string stream
        
        // do the same for negative sample:
        os << TRAINING_SET_PATH << "negative\\" << std::setw(4) << std::setfill('0') << i << ".JPG";
        img = imread(os.str(),CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data)
            break;
        
        hog.compute(img, featureVector, Size(8, 8), Size(0, 0));
        svm.writeFeatureVectorToFile(featureVector, false);
        negCount++;
        img.release();
    }

    std::cout   << "finished writing features: "
                << posCount << " positive and "
                << negCount << " negative samples used";
    std::string modelName("classifier.dat");
    svm.trainAndSaveModel(modelName);
    std::cout   << "SVM saved to " << modelName;
```

#####Classification

The classification can be then performed by the instance of `HOGDescriptor`, you just need to feed it with your own SVM detector in form of `std::vector<float>`:

```C++
    HOGDescriptor hog;
    hog.winSize = Size(32,48);
    SVMLight::SVMClassifier c("classifier.dat");
    vector<float> descriptorVector = c.getDescriptorVector();
    hog.setSVMDetector(descriptorVector);
```

and later once you retrieve some segment, you can do something like:

```C++
    Mat segment = img(Rect(x0, y0, x1 - x0, y1 - y0));
    vector<Rect> found;
    Size padding(Size(0, 0));
    Size winStride(Size(8, 8));
    hog.detectMultiScale(segment, found, 0.0, winStride, padding, 1.01, 0.1);
```

That's all :)
