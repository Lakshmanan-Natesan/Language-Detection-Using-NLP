{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "app.py",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOF3/4bxRvsIdtF147dIgYK",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Lakshmanan-Natesan/Language-Detection-Using-NLP/blob/master/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 373
        },
        "id": "raZVsQ44wcwl",
        "outputId": "9074b54e-7bfa-434b-852d-01cdf348157f"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ImportError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-2-035828efd1d6>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfeature_extraction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtext\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mCountVectorizer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnaive_bayes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mMultinomialNB\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexternals\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mjoblib\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpickle\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mImportError\u001b[0m: cannot import name 'joblib' from 'sklearn.externals' (/usr/local/lib/python3.7/dist-packages/sklearn/externals/__init__.py)",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "from flask import Flask,render_template,url_for,request\n",
        "import pandas as pd \n",
        "import pickle\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.externals import joblib\n",
        "import pickle\n",
        "\n",
        "# load the model from disk\n",
        "filename = 'nlp_model.pkl'\n",
        "clf = pickle.load(open(filename, 'rb'))\n",
        "cv=pickle.load(open('tranform.pkl','rb'))\n",
        "app = Flask(__name__)\n",
        "\n",
        "@app.route('/')\n",
        "def home():\n",
        "\treturn render_template('home.html')\n",
        "\n",
        "@app.route('/predict',methods=['POST'])\n",
        "def predict():\n",
        "#\tdf= pd.read_csv(\"spam.csv\", encoding=\"latin-1\")\n",
        "#\tdf.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)\n",
        "#\t# Features and Labels\n",
        "#\tdf['label'] = df['class'].map({'ham': 0, 'spam': 1})\n",
        "#\tX = df['message']\n",
        "#\ty = df['label']\n",
        "#\t\n",
        "#\t# Extract Feature With CountVectorizer\n",
        "#\tcv = CountVectorizer()\n",
        "#\tX = cv.fit_transform(X) # Fit the Data\n",
        "#    \n",
        "#    pickle.dump(cv, open('tranform.pkl', 'wb'))\n",
        "#    \n",
        "#    \n",
        "#\tfrom sklearn.model_selection import train_test_split\n",
        "#\tX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n",
        "#\t#Naive Bayes Classifier\n",
        "#\tfrom sklearn.naive_bayes import MultinomialNB\n",
        "#\n",
        "#\tclf = MultinomialNB()\n",
        "#\tclf.fit(X_train,y_train)\n",
        "#\tclf.score(X_test,y_test)\n",
        "#    filename = 'nlp_model.pkl'\n",
        "#    pickle.dump(clf, open(filename, 'wb'))\n",
        "    \n",
        "\t#Alternative Usage of Saved Model\n",
        "\t# joblib.dump(clf, 'NB_spam_model.pkl')\n",
        "\t# NB_spam_model = open('NB_spam_model.pkl','rb')\n",
        "\t# clf = joblib.load(NB_spam_model)\n",
        "\n",
        "\tif request.method == 'POST':\n",
        "\t\tmessage = request.form['message']\n",
        "\t\tdata = [message]\n",
        "\t\tvect = cv.transform(data).toarray()\n",
        "\t\tmy_prediction = clf.predict(vect)\n",
        "\treturn render_template('result.html',prediction = my_prediction)\n",
        "\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "\tapp.run(debug=True)\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "veOzIXQpweRY"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}