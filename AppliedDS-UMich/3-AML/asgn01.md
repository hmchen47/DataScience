# Assignment 1

## Notebook Links

+ [Launching Web Page](https://www.coursera.org/learn/python-machine-learning/notebook/oxndk/assignment-1)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=8yJcS-rfTAeiXEvq3_wHmg&next=%2Fnotebooks%2FAssignment%25201.ipynb)
+ [Local Notebook](notebooks/Assignment01.ipynb)

## Useful Info Links

### [Assignment 1 FAQs](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/1/threads/KG7q2GIhEeeg5QoVkEgmCg)

+ Q1 Basic Check script

    ```python
    #test 
    def test_one():
        columns =cancer.feature_names.tolist() +['target']
        index =pd.RangeIndex(start=0, stop=569, step=1).tolist()
        df = answer_one()
        res = 'Type Test: '
        res += ['Failed\n','Passed\n'][type(df)==pd.DataFrame]
        
        res += 'dtypes Test: '
        res += ['Failed; all columns should have float64(last column can be int64) \n',
                'Passed\n'][all(df[df.columns[:-1]].dtypes=='float64') and df[df.columns[-1]].dtype in ['float64','int64']]
        
        res += 'df shape Test: '
        res += ['Failed\n','Passed\n'][df.shape==(569,31)]

        res += 'Columns Names Test: '
        res +=  ['Failed\n','Passed\n'][all(df.columns == columns)]

        res += 'Index Test: '
        res +=  ['Failed\n','Passed\n'][df.index.tolist()==index]
        
        res += 'Values test: '
        res +=  ['Failed\n','Passed\n'][(df[df.columns[:-1]].values==cancer.data).all().all()]
        
        res += 'target test: '
        try:
            res +=  ['Failed\n','Passed\n'][(df[df.columns[-1]].values*1.0==cancer.target*1.0).all()]
        except:
            res += 'Falied: target must be numercial\n'
        return res
    print(test_one())
    ```

+ the first step will be to look at the grader output, you can view the grader output in your "My submission page" by clicking on the latest submission and clicking "show grader output"

    scrolling down the output you will be able to see the result of grading each question. or an error message

    Check the error message. the notebook is graded as script, i.e. all the cells in the notebook need to run without error for the grading to take place.
    
    to find the issue, go to your notebook and click Kernel->Restart and Run all. check the output of each cell, and fix any issues, save and resubmit.

    I've explained a bit more about how the grader works and best way to debug/troubleshoot, [here](https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng)

+ There is an indentation problem that caused your notebook to not be converted to python script. you'll see the error if you try to download the notebook as python .py file
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/weeks/1/threads/KG7q2GIhEeeg5QoVkEgmCg">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Pz818HqsEeeOygpRbdVQKg_8951cea0acb8aab84a422d2cfc95594f_Screen-Shot-2017-08-06-at-14.36.08.png?expiry=1535760000000&hmac=Cx1IXzT7COXfnnLchEbgU9uUCMVl7NcETDky_TqCE0Q" alt="text" title= "iPython File Menu" width="450">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/weeks/1/threads/KG7q2GIhEeeg5QoVkEgmCg">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/S-bN4nqsEeeOygpRbdVQKg_b34151568a5ccd90fab20aa3e04d76ad_Screen-Shot-2017-08-06-at-14.35.10.png?expiry=1535760000000&hmac=Z9HRBrPub_mPMiz8om5D1bc-Emd5XQGhixKr9Llv-pI" alt="500: Internal Server Error" title= "caption" width="450">
    </a>

    finally, I found its the first line in answer_two(), make sure the indentation of the first line is the same as the rest of the function,


+ Q3 test

    ```python
    #test 
    def test_three():
        columnsX =cancer.feature_names
        shapeX = cancer.data.shape
        namey ='target'
        shapey = cancer.target.shape
        X, y = answer_three()
        
        res  = 'X type Test: '
        res +=  ['Failed\n','Passed\n'][type(X)==pd.DataFrame]
        
        res += 'y type Test: '
        res +=  ['Failed\n','Passed\n'][type(y)==pd.Series]

        
        res += 'X shape Test: '
        res += ['Failed\n','Passed\n'][int(X.shape==shapeX)]

        res += 'y shape Test: '
        res +=  ['Failed\n','Passed\n'][int(y.shape==shapey)]
        try:
            res += 'X columns Test: '
            res +=  ['Failed\n','Passed\n'][int(all(X.columns==columnsX))]
    
            res += 'y name Test: '
            res +=  ['Failed\n','Passed\n'][int(y.name==namey)]
    
            res += 'X data Test: '
            res +=  ['Failed\n','Passed\n'][int(all(X.values.reshape(1,-1)[0]==cancer.data.reshape(1,-1)[0]))]
    
            res += 'y data Test: '
            res +=  ['Failed\n','Passed\n'][int(all(y.values==cancer.target))]

            res += 'X columns length Test: '
            res +=  ['Failed\n','Passed\n'][X.shape[1] ==len(cancer.feature_names)]

            res += 'X column names Test: '
            res +=  ['Failed\n','Passed\n'][all(X.columns==cancer.feature_names)]     
        except:
            print('Data type problem, X should be a pandas DataFrame and y should be a pandas Series')
            return res
        return res

    print(test_three())
    ```





### [More on the Grader](https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng)

+ Before submitting the assignment notebook in this course/specialisation, comment/delete the following
    + any calls to any plot functions precoded in the assignment or any plot code you write yourself
    + any magic functions (anything starting with %)
    + any unassigned linux commands (anything that starts with !),
    + any global import statements of __matplotlib, seaborn or adspy_shared_utilities__

+ A word of advice about the grader; a lot of issues and confusion arise due to difference between the interactive nature of notebooks and how you run them as an author and the way the grader executes the code; i.e.. notebook converted to python script then script is run then the functions are called one by one for evaluation. The best way to catch these issues is to run the notebook in a way that simulate how the grader works; i.e. test the functions in the last cell of the notebook rather than after each function definition and also Run the notebook with a fresh kernel. here is how I test my notebooks before I submit them for grading

    ```python
    fDict =globals()
    res = {k:v() for k,v in fDict.items() if k.startswith('answer')}
    res['answer_...']
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/JObvSVPwEeeroBJvqbisdg_5218f5c89da9d846b5689e501dca3ae1_Screen-Shot-2017-06-18-at-07.32.27.png?expiry=1535760000000&hmac=tcA-3-SG0Me0Kr1j6Qq1q3kXZNnmEfCqnmCK3_RnSs4" alt="Inspection" title= "caption" width="450">
    </a>

    and then I inspect individual functions, e.g.
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ROIJvFPwEeeroBJvqbisdg_96f5c564855119bf4a98c17cd76c7f63_Screen-Shot-2017-06-18-at-07.33.43.png?expiry=1535760000000&hmac=4xqocnGItKTjleojh7o9IVtiPbBeJ96yAUwcudwYtzg" alt="Inspection" title= "caption" width="450">
    </a>


























