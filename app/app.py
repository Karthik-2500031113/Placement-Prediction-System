from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset")
def dataset():
    return "<h2>Dataset Page</h2>"

@app.route("/eda")
def eda():
    return "<h2>EDA Page</h2>"

@app.route("/preprocessing")
def preprocessing():
    return "<h2>Preprocessing Page</h2>"

@app.route("/models")
def models():
    return "<h2>Models Page</h2>"

@app.route("/results")
def results():
    return "<h2>Results Page</h2>"

if __name__ == "__main__":
    app.run(debug=True)