import os
import logging

from flask import Flask, request , render_template

from Emotion_Fluency_Model import Models_Emotions_Fluency

app = Flask(__name__)

# define model path
model_emotion = 'model_Emotions.h5'
model_fluency = 'model_Fluency.h5'


# create instance
model = Models_Emotions_Fluency(model_emotion,model_fluency)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    """Provide simple health check route."""
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")

    inputted_text = [i for i in request.form.values()]

    pred_emotion, pred_fluency, average_fluency = model.predict(inputted_text[0],5)




    fluency_prediction_text = ["Your Flwency in the uploaded video was as follows:" ,
                              f"{pred_fluency['Low']}% of your pitching was Very Low",
                              f"{pred_fluency['Intermediate']}% of your pitching was Intermediate",
                              f"{pred_fluency['High']}% of your pitching was High"]

    emotion_prediction_text = ["Your Emotions in the uploaded video was as follows:",
                              f"{pred_emotion['sleepy']}% of your pitching was sleepy",
                              f"{pred_emotion['neutral']}% of your pitching was neutral",
                              f"{pred_emotion['happy']}% of your pitching was happy",
                              f"{pred_emotion['angry']}% of your pitching was Very angry" ,
                              f"{pred_emotion['disgusted']}% of your pitching was disgusted",
                              f"{pred_emotion['fear']}% of your pitching was fear",
                              f"{pred_emotion['surprise']}% of your pitching was surprise"]

    #logging.info("prediction from model= {}".format(prediction))
    logging.info(pred_fluency,pred_emotion)


    return render_template("index.html", fluency_prediction_text_0 = fluency_prediction_text[0],
                                        fluency_prediction_text_1 = fluency_prediction_text[1],
                                        fluency_prediction_text_2 = fluency_prediction_text[2],
                                        fluency_prediction_text_3 = fluency_prediction_text[3],

                                        emotion_prediction_text_0 =emotion_prediction_text[0],
                                        emotion_prediction_text_1 = emotion_prediction_text[1],
                                        emotion_prediction_text_2 = emotion_prediction_text[2],
                                        emotion_prediction_text_3 = emotion_prediction_text[3],
                                        emotion_prediction_text_4 = emotion_prediction_text[4],
                                        emotion_prediction_text_5 = emotion_prediction_text[5],
                                       emotion_prediction_text_6 = emotion_prediction_text[6],
                                       emotion_prediction_text_7 = emotion_prediction_text[7] )



def main():
    """Run the Flask app."""
    port = os.environ.get("PORT", 5500)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()