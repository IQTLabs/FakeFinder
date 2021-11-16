## API Class Diagram

![](classes_FakeFinder.png?raw=true)

## Using the API layer

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) POST INFERENCE - To run inference, send a Post request to - ``` http://<ip address>:5000/fakefinder ```

The Post inference request supports 2 modes - UI and Batch.  The GPU-powered AWS EC2 instances can be expensive to run indefinitely. On the other hand, bringing up an GPU EC2 instance from an image in an on-demand scenario takes some time.  In view of these considerations, the API provides two modes that support either using AlwaysON or Warm (Stopped) instances, and Cold start(newly created from image) instances.  The first of these is the UI mode and provides significantly better response time than second one which is the batch mode.

The inference request model is shown below - 

```
FakeFinder{
batchMode*	boolean
default: false
Set this field to true if processing video/image files in batch mode. If requests are coming from a UI this should be set to false.

alwaysOn	boolean
default: true
Set this field to true if starting/stopping ec2 instances

location*	string
Image/Video location.

modelName*	string
Name of the model to run inference against

splitRequests	boolean
default: false
Split the request containing a list of videos to multiple requests containing single video.

numSplitRequests	integer
default: 1
Number of splits of the list containing videos.

}
```

### UI Mode

In this mode "batchMode" is set to False.  There is an option of running inference against running or stopped instances by toggling the "AlwaysOn" attribute. If this attribute is set to "True" the API expects running ec2 instances to which inference request can be sent. If this is set to False, the API will start the inference server corresponding to the "modelName" attribute first before running the inference. The later scenario will have a longer response time compared to the first one. A request containing long list of files can also be split into multiple requests, however, the reponse will always be an aggregate of all the requests.

### Batch Mode

In this mode "AlwaysOn" is set to False. API always creates an EC2 instance from the image stored in the container registery. The image is selected based on the "modelName" parameter. There is an option to split a request containing large number of files to different ec2 instances. If split request is selected, the API divides the list of files, spawns ec2 instances based on "numSplitRequests" and sends different requests to each of the instances. This provides the scaling needed for large scale inferencing scenarios. The reponse will always be an aggregate of all the requests.

![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) GET MODELS - To get a list of available models, send a Get request to - ``` http://<ip address>:5000/fakefinder ```

The API is fully configurable and can support any number of inferencing models. A new model can be added, swapped or deleted as described in earlier sections. At any given time, a list of supported models can be obtained by sending a Get request.

![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) POST AWS S3 UPLOAD - To upload a file to S3 bucket before inference, send a request to - ``` http://<ip address>:5000/upload ```

The API also supports uploading a file to S3 bucket before making an inferencing request.

### Future Work

The API does not currently support return partial responses when a large request is split into multiple requests. This can be done by implementing a messaging queue. The API can then publish interim responses before returning with the aggregate response.
