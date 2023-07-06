from classifier import classify_text
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        query = request.form['query']
        category, probabilities = classify_text(query)
        probabilities = op(probabilities)
        return render_template('index.html', query=query, category=category, probabilities=probabilities)
    
    return render_template('index.html', query="", category="", probabilities=[])

def op(pr):
    BUSINESS = round(pr[0][0], 3)
    SPORT = round(pr[0][1], 3)
    HEALTH = round(pr[0][2], 3)

    lines = [BUSINESS, SPORT, HEALTH]
    return lines

# driver function
if __name__ == '__main__':
    app.run(debug=True)
