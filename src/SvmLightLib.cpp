#include <vector>
#include <ios>
#include <string>
#include <fstream>
#include "SvmLightLib.h"

/**
 * wrapper of wrapper of SVMLight by LihO
 * based on projects: 
 *    http://mihagrcar.org/svmlightlib/ 
 *    https://github.com/DaHoC/trainHOG
 * 
 */

namespace SVMLight 
{
	extern "C" 
	{
		#include "svm_common.h"	
        #include "svm_learn.h"
	}

    // more or less original wrapper of SVMLight taken from https://github.com/DaHoC/trainHOG
    class SVMLightImpl {
    private:
        DOC** docs; // training examples
        long totwords, totdoc, i; // support vector stuff
        double* target;
        double* alpha_in;
        KERNEL_CACHE* kernel_cache;
        MODEL* model; // SVM model

        SVMLightImpl() {
            // Init variables
            alpha_in = NULL;
            kernel_cache = NULL; // Cache not needed with linear kernel
            model = (MODEL *) my_malloc(sizeof (MODEL));
            learn_parm = new LEARN_PARM;
            kernel_parm = new KERNEL_PARM;
            // Init parameters
            verbosity = 1; // Show some messages -v 1
            learn_parm->alphafile[0] = NULL;
            learn_parm->biased_hyperplane = 1;
            learn_parm->sharedslack = 0; // 1
            learn_parm->skip_final_opt_check = 0;
            learn_parm->svm_maxqpsize = 10;
            learn_parm->svm_newvarsinqp = 0;
            learn_parm->svm_iter_to_shrink = 2; // 2 is for linear;
            learn_parm->kernel_cache_size = 40;
            learn_parm->maxiter = 100000;
            learn_parm->svm_costratio = 1.0;
            learn_parm->svm_costratio_unlab = 1.0;
            learn_parm->svm_unlabbound = 1E-5;
            learn_parm->eps = 0.1;
            learn_parm->transduction_posratio = -1.0;
            learn_parm->epsilon_crit = 0.001;
            learn_parm->epsilon_a = 1E-15;
            learn_parm->compute_loo = 0;
            learn_parm->rho = 1.0;
            learn_parm->xa_depth = 0;
            // The HOG paper uses a soft classifier (C = 0.01), set to 0.0 to get the default calculation
            learn_parm->svm_c = 0.01; // -c 0.01
            learn_parm->type = CLASSIFICATION;
            learn_parm->remove_inconsistent = 0; // -i 0 - Important
            kernel_parm->rbf_gamma = 1.0;
            kernel_parm->coef_lin = 1;
            kernel_parm->coef_const = 1;
            kernel_parm->kernel_type = LINEAR; // -t 0
            kernel_parm->poly_degree = 3;
			
			kernel_parm->custom[0] = '0';
			kernel_parm->custom[1] = '\0';
        }

        virtual ~SVMLightImpl() {
            // Cleanup area
            // Free the memory used for the cache
            if (kernel_cache)
                kernel_cache_cleanup(kernel_cache);
            free(alpha_in);
            free_model(model, 0);
            for (i = 0; i < totdoc; i++)
                free_example(docs[i], 1);
            free(docs);
            free(target);
        }

    public:
        LEARN_PARM* learn_parm;
        KERNEL_PARM* kernel_parm;

        static SVMLightImpl* getInstance() {
            static SVMLightImpl theInstance;
            return &theInstance;
        }

        inline void saveModelToFile(const std::string& _modelFileName) {
            write_model(const_cast<char*>(_modelFileName.c_str()), model);
        }

        void loadModelFromFile(const std::string& _modelFileName) {
            this->model = read_model(const_cast<char*>(_modelFileName.c_str()));
        }

        // read in a problem (in SVMLightImpl format)
        void read_problem(const std::string& filename) {
            // Reads and parses the specified file
            read_documents(const_cast<char*> (filename.c_str()), &docs, &target, &totwords, &totdoc);
        }

        // Calls the actual machine learning algorithm
        void train() {
            svm_learn_classification(docs, target, totdoc, totwords, learn_parm, kernel_parm, kernel_cache, model, NULL, create_env());
            // original wrapper was using the regression:
            // svm_learn_regression(docs, target, totdoc, totwords, learn_parm, kernel_parm, &kernel_cache, model, create_env());
        }

        /**
         * Generates a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
         * vec1 = sum_1_n (alpha_y*x_i). (vec1 is a 1 x n column vector. n = feature vector length )
         * @param singleDetectorVector resulting single detector vector for use in openCV HOG
         * @param singleDetectorVectorIndices
         */
        void getSingleDetectingVector(std::vector<float>& singleDetectorVector, std::vector<unsigned int>& singleDetectorVectorIndices) {
            // Now we use the trained svm to retrieve the single detector vector
            DOC** supveclist = model->supvec;
            printf("Calculating single descriptor vector out of support vectors (may take some time)\n");
            // Retrieve single detecting vector (v1 | b) from returned ones by calculating vec1 = sum_1_n (alpha_y*x_i). (vec1 is a n x1 column vector. n = feature vector length )
            singleDetectorVector.clear();
            singleDetectorVector.resize(model->totwords + 1, 0.);
            printf("Resulting vector size %lu\n", singleDetectorVector.size());
        
            // Walk over every support vector
            for (long ssv = 1; ssv < model->sv_num; ++ssv) { // Don't know what's inside model->supvec[0] ?!
                // Get a single support vector
                DOC* singleSupportVector = supveclist[ssv]; // Get next support vector
                SVECTOR* singleSupportVectorValues = singleSupportVector->fvec;
                WORD singleSupportVectorComponent;
                // Walk through components of the support vector and populate our detector vector
                for (long singleFeature = 0; singleFeature < model->totwords; ++singleFeature) {
                    singleSupportVectorComponent = singleSupportVectorValues->words[singleFeature];
                    singleDetectorVector.at(singleSupportVectorComponent.wnum) += (singleSupportVectorComponent.weight * model->alpha[ssv]);
                }
            }

            // This is a threshold value which is also recorded in the lear code in lib/windetect.cpp at line 1297 as linearbias and in the original paper as constant epsilon, but no comment on how it is generated
            singleDetectorVector.at(model->totwords) = -model->b; /** @NOTE the minus sign! */
        }

    };

    // SVMTrainer & SVMClassifier implementations:

    SVMTrainer::SVMTrainer(const std::string& featuresFileName)
    {
        // use the C locale while creating the model file:
        setlocale(LC_ALL, "C");

        featuresFileName_ = featuresFileName;
        featuresFile_.open(featuresFileName_.c_str(), std::ios::out);
    }

    void SVMTrainer::writeFeatureVectorToFile(const std::vector<float>& featureVector, bool isPositive)
    {
        featuresFile_ << ((isPositive) ? "+1" : "-1");
        for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
            featuresFile_ << " " << (feature + 1) << ":" << featureVector.at(feature);
        }
        featuresFile_ << std::endl;
    }

    void SVMTrainer::trainAndSaveModel(const std::string& modelFileName)
    {
        if (featuresFile_.is_open())
            featuresFile_.close();

        SVMLightImpl::getInstance()->read_problem(featuresFileName_);
        SVMLightImpl::getInstance()->train();
        SVMLightImpl::getInstance()->saveModelToFile(modelFileName);
    }

    SVMClassifier::SVMClassifier(const std::string& modelFilename)
    {
         SVMLightImpl::getInstance()->loadModelFromFile(modelFilename);
    }
        
    std::vector<float> SVMClassifier::getDescriptorVector()
    {
        std::vector<float> descriptorVector;       
        SVMLightImpl::getInstance()->getSingleDetectingVector(descriptorVector, std::vector<unsigned int>() /* indices */);
        return descriptorVector;
    }

}
