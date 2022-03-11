import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from flask_ngrok import run_with_ngrok
from flask_restful import Resource, Api
from flask_cors import CORS
from flask import Flask, render_template , request 
import os
from google.colab import drive

app = Flask(__name__)
api = Api(app)
CORS(app)
run_with_ngrok(app)
class status (Resource):
    def get(self):
        try:
              text=["""JGears are toothed, mechanical transmission elements used to transfer motion and power between machine components, and in this article, we discuss the different types of gears available and how gears work. Operating in mated pairs, gears mesh their teeth with the teeth of another corresponding gear or toothed component which prevents slippage during the transmission process. Each gear or toothed component is attached to a machine shaft or base component, therefore when the driving gear (i.e., the gear that provides the initial rotational input) rotates along with its shaft component, the driven gear (i.e., the gear or toothed component which is impacted by the driving gear and exhibits the final output) rotates or translates its shaft component. Depending on the design and construction of the gear pair, the transference of motion between the driving shaft and the driven shaft can result in a change of the direction of rotation or movement. Additionally, if the gears are not of equal sizes, the machine or system experiences a mechanical advantage which allows for a change in the output speed and torque (i.e., the force which causes an object to rotate). Gears and their mechanical characteristics are widely employed throughout industry to transmit motion and power in a variety of mechanical devices, such as clocks, instrumentation, and equipment, and to reduce or increase speed and torque in a variety of motorized devices, including automobiles, motorcycles, and machines. Other design characteristics, including construction material, gear shape, tooth construction and design, and gear pair configuration, help to classify and categorize the various types of gears available. Each of these gears offers different behaviors and advantages, but the requirements and specifications demanded by a particular motion or power transmission application determine the type of gear most suitable for use."""]
              model_name = 'google/pegasus-xsum'

              torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
              tokenizer = PegasusTokenizer.from_pretrained(model_name)
              model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
              batch = tokenizer.prepare_seq2seq_batch(text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
              translated = model.generate(**batch)
              t_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
              return {'data':t_text[0]}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

api.add_resource(status, '/')

if __name__ == '__main__':
   app.run()