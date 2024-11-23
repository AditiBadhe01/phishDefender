import falcon
from training_service import TrainModelResource

# Initialize Falcon app
app = falcon.App()

# Add routes
app.add_route('/train-model', TrainModelResource())


