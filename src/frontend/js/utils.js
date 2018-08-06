var service = new proto.feedhandling.FeedHandlingClient('http://' + window.location.hostname + ':8888');

$("#retrain_btn").click(function() {
    var train_request = new proto.feedhandling.Empty();

    service.initiateTraining(train_request, {}, function(err, response) {
        console.log(response)
    });
});

