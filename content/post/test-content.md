---
title: "{{ replace .TranslationBaseName "-" " " | title }}"
date: {{ .Date }}
lastmod: {{ .Date }}
draft: false
keywords: []
description: ""
tags: []
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false
toc: false
autoCollapseToc: false
postMetaInFooter: false
hiddenFromHomePage: false
# You can also define another contentCopyright. e.g. contentCopyright: "This is another copyright."
contentCopyright: false
reward: false
mathjax: false
mathjaxEnableSingleDollar: false
mathjaxEnableAutoNumber: false

# You unlisted posts you might want not want the header or footer to show
hideHeaderAndFooter: false

# You can enable or disable out-of-date content warning for individual post.
# Comment this out to use the global config.
#enableOutdatedInfoWarning: false

flowchartDiagrams:
  enable: false
  options: ""

sequenceDiagrams: 
  enable: false
  options: ""

---

<!--more-->


# Model deployment with Tensorflow serving & FastAPI
### Prepare
In your terminal run the following command below:  
`docker pull tensorflow/serving`

Save model you want to predict:
```python
export_path = "/content/drive/MyDrive/Kaggle/HappyWhale-2022/predict_model_v3"  
tf.keras.models.save_model(
    model,
    export_path
)
```
The model will be save with the following structure:
![structure](/media/model_structure.JPG)

### Serving saved model with Tensorflow Serving
Bind source with target (model): 

`docker run --rm -p 8501:8501 --mount type=bind,source="$(pwd)"/predict_model_embed,target=/models/predict_model_embed -e MODEL_NAME=predict_model_embed -t tensorflow/serving` 

Explain:  
    **-p 8501:8501**: This is the REST Endpoint port. Every prediction request will be made to this port. For instance, you can make a prediction request to http://localhost:8501.  
    **--name tfserving_classifier**: This is a name given to the Docker container running TF Serving. It can be used to start and stop the container instance later.   
    **-- mount type=bind,source=/Users/tf-server/img_classifier/,target=/models/img_classifier**: The mount command simply copies the model from the specified path (**/Users/tf-server/img_classifier/**) into the Docker container (**/models/img_classifier**), so that TF Serving has access to it.  
    **-e MODEL_NAME=img_classifier**: The name of the model to run. This is the name you used to save your model.
    **-t tensorflow/serving**: The TF Serving Docker container to run.  

Next, define the REST Endpoint URL:  
`url = 'http://localhost:8501/v1/models/img_classifier:predict'`  
The prediction URL is made up of a few important parts. A general structure may look like the one below:  

`http://{HOST}:{PORT}/v1/models/{MODEL_NAME}:{VERB}`

**HOST**: The domain name or IP address of your model server  
**PORT**: The server port for your URL. By default, TF Serving uses 8501 for REST Endpoint.  
**MODEL_NAME**: The name of the model you’re serving.  
**VERB**: The verb has to do with your model signature. You can specify one of predict, classify or regress.  

Next, add a function to make a request to Endpoint:

```python
def make_prediction(instances):
   data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   predictions = json.loads(json_response.text)
   return predictions
```
### Create server with FastAPI to interact with tensorflow serving
```python
app = FastAPI(title='Deploying a ML Model with FastAPI')

#using @app.get("/") allow Get method woking with / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."


#Endpoint use for prediction
@app.post("/predict") 
def prediction(file: UploadFile = File(...)):

    #1. Verify input file
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    #2. Convert from raw to CV2
    
    #Read byte stream
    image_stream = io.BytesIO(file.file.read())
    
    #Start from beginning
    image_stream.seek(0)
    
    #Cast to numpy type
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    #Resize image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image,(768,768),interpolation=cv2.INTER_AREA)
    image = image/255.0
    image = np.expand_dims(image,axis=0)
    #3. Run prediction model
    embedding=make_prediction(image)['predictions']
    distances,idxs = neigh.kneighbors(embedding, 100, return_distance=True)
    test_nn_idxs=idxs
    test_nn_distances=distances
    test_target=[train_targets[i] for i in test_nn_idxs[0]]
    test_df=pd.DataFrame({'target':test_target,'distance':test_nn_distances[0]})
    test_df['confidence']=1-test_df['distance']
    test_df=test_df.groupby(['target']).confidence.max().reset_index()
    test_df=test_df.sort_values('confidence',ascending=False)
    test_df['target']=test_df['target'].map(target_encodings)

    prediction={'ID':[],'Species':[]}
    for i,row in test_df.iterrows():
        if len(prediction)==5:
            continue
        else:
            if (row.confidence<0.56 and len(prediction)==0):
                prediction['ID'].extend(('new_individual',row.target))
            else:
                prediction['ID'].append(row.target)
    with open('images_uploaded/predict.txt','w') as file:
        file.write(' '.join([str(x) for x in prediction]))
    file=open('images_uploaded/predict.txt',mode='rb')
    # Return prediction in text file 
    return StreamingResponse(file,media_type='text/html')
```

### Running server  

```python
#Allow server running in this environment
nest_asyncio.apply()
#Specify host address
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"
#Running server!    
uvicorn.run(app, host=host, port=8000)
```

Now we ready to make prediction at: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

![model_api](/media/model_fast_api.JPG)
![model_api_2](/media/model_fast_api_2.JPG)