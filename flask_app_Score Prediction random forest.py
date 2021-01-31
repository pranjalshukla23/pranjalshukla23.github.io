#!/usr/bin/env python
# coding: utf-8

# In[1]:



# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request
from Score_prediction_rf import predict
from flask import render_template


app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods=["GET", "POST"])
def adder_page():
    errors = ""
    if request.method == "POST":
        number1=None
        number2=None
        number3=None
        number4=None
        number5=None
        try:
            number1 = int(request.form["runs"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["runs"])
        try:
            number2 = int(request.form["wickets"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["wickets"])
        try:
            number3 = float(request.form["overs"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["overs"])
        try:
            number4 = int(request.form["striker"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["striker"])
        try:
            number5 = int(request.form["non_striker"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["non_striker"])
            

        
        if number1 is not None and number2 is not None and number3 is not None and number4 is not None and number5 is not None:
            result = predict(number1,number2,number3,number4,number5)
            return '''
                <html>
                    <body>
                        <p>The result is {result}</p>
                        <p><a href="/">Click here to calculate again</a><br>
                       </body>
                </html>
            '''.format(result=result)

    return '''
        <html>
            <body>
                {errors}
                <p>Enter your numbers:</p>
                <form method="post" action=".">
                    <label for="runs">Enter the runs</label>
                    <input type="text" name="runs"/><br>
                    <label for="wickets">Enter the Wickets:</label>
                    <input name="wickets" type="text"/><br>
                    <label for="overs">Enter the overs:</label>
                    <input name="overs" type="text"/><br>
                    <label for="striker">Enter the striker runs:</label>
                    <input name="striker" type="text" /><br>
                    <label for="non-striker">Enter the non-striker runs:</label>
                    <input name="non_striker" type="text"/>
                    <p><input type="submit" value="Predict the score" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)
if __name__== "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 10000, app)


# In[ ]:




