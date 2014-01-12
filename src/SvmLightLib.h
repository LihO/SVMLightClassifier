#ifndef __SVMLIGHTLIB_H__
#define __SVMLIGHTLIB_H__

#include <vector>
#include <string>
#include <fstream>

namespace SVMLight
{
    extern class SVMTrainer
    {
    private:
        std::fstream featuresFile_;
        std::string featuresFileName_;
    public:
        SVMTrainer(const std::string& featuresFileName);
        void writeFeatureVectorToFile(const std::vector<float>& featureVector, bool isPositive);
        void trainAndSaveModel(std::string& modelFileName);
    };

    extern class SVMClassifier
    {
    public:
        SVMClassifier(std::string& featuresFileName);
        std::vector<float> getDescriptorVector();
    };
}

#endif
