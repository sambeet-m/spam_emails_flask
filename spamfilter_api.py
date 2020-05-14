from flask import render_template, request, flash, redirect, Blueprint, url_for
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
import os, re
from flask import current_app
from spamfilter.models import db, File
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from shutil import copyfile

from spamfilter.forms import InputForm
from spamfilter import spamclassifier

spam_api = Blueprint('SpamAPI', __name__)

def allowed_file(filename, extensions=None):
    '''
    'extensions' is either None or a list of file extensions.

    If a list is passed as 'extensions' argument, check if 'filename' contains
    one of the extension provided in the list and return True or False respectively.

    If no list is passed to 'extensions' argument, then check if 'filename' contains
    one of the extension provided in list 'ALLOWED_EXTENSIONS', defined in 'config.py',
    and return True or False respectively.
    '''
    name, ext = os.path.splitext(filename)
    print("In allowed_file()", name, ext, extensions )
    #print("In allowed_file() 1", extensions, ALLOWED_EXTENSIONS )
    if extensions is None:
      print("In allowed_file() 2", name, ext, extensions )
      if ext in ALLOWED_EXTENSIONS:
        return True
      else:
        return False
    else:
      if ext in extensions:
        return True
      else:
        return False


@spam_api.route('/')
def index():
    '''
    Renders 'index.html'
    '''
    return render_template('index.html')

@spam_api.route('/listfiles/<success_file>/')
@spam_api.route('/listfiles/')
def display_files(success_file=None):
    '''
    Obtain the filenames of all CSV files present in 'inputdata' folder and
    pass it to template variable 'files'.

    Renders 'filelist.html' template with values  of variables 'files' and 'fname'.
    'fname' is set to value of 'success_file' argument.

    if 'success_file' value is passed, corresponding file is highlighted.
    '''
    print("In display_files() ", os.getcwd())
    Input_directory = os.path.join(os.path.join(os.getcwd(), 'spamfilter/inputdata/'))
    ListOfFiles = os.listdir(Input_directory)
    ListOfCsvFiles = []
    for file in ListOfFiles:
      name, ext = os.path.splitext(file)
      if ext == '.csv':
        ListOfCsvFiles.append(file)
    print("In display_files() ", ListOfFiles,  ListOfCsvFiles,Input_directory)
    return render_template('fileslist.html', files=ListOfCsvFiles, fname=success_file )



def validate_input_dataset(input_dataset_path):
    '''
    Validate the following details of an Uploaded CSV file

    1. The CSV file must contain only 2 columns. If not display the below error message.
    'Only 2 columns allowed: Your input csv file has '+<No_of_Columns_found>+ ' number of columns.'

    2. The column names must be "text" nad "spam" only. If not display the below error message.
    'Differnt Column Names: Only column names "text" and "spam" are allowed.'

    3. The 'spam' column must conatin only integers. If not display the below error message.
    'Values of spam column are not of integer type.'

    4. The values of 'spam' must be either 0 or 1. If not display the below error message.
    'Only 1 and 0 values are allowed in spam column: Unwanted values ' + <Unwanted values joined by comma> + ' appear in spam column'

    5. The 'text' column must contain string values. If not display the below error message.
    'Values of text column are not of string type.'

    6. Every input email must start with 'Subject:' pattern. If not display the below error message.
    'Some of the input emails does not start with keyword "Subject:".'

    Return False if any of the above 6 validations fail.

    Return True if all 6 validations pass.
    '''
    print("In validate_input_dataset()", input_dataset_path)
    df = pd.read_csv(input_dataset_path)
    #1
    if len(df.columns) != 2:
        error = 'Only 2 columns allowed: Your input csv file has '+ str(df.columns) + ' number of columns.'
        flash(error)
        return False

    #2
    for col in df.columns:
        if col in ['text', 'spam']:
            pass
        else:
            flash('Differnt Column Names: Only column names "text" and "spam" are allowed.')
            return False

    if str(df['spam'].dtype)[0:3] != 'int':
        flash('Values of spam column are not of integer type.')
        return False

    if set(df['spam']) != {0,1}:
        unwantedValue = str(list (set(df['spam']).difference({0,1})))[1:-1]
        error = 'Only 1 and 0 values are allowed in spam column: Unwanted values ' + unwantedValue + ' appear in spam column'
        flash(error)
        return False

	  #5
    if df['text'].dtype != 'O':
        flash('Values of text column are not of string type.')
        return False

    #6
    if set(df['text'].str.startswith('Subject:')) != {True}:
        flash('Some of the input emails does not start with keyword "Subject:".')
        return False

    return True



@spam_api.route('/upload/', methods=['GET', 'POST'])
def file_upload():
    '''
    If request is GET, Render 'upload.html'

    If request is POST, capture the uploaded file a

    check if the uploaded file is 'csv' extension, using 'allowed_file' defined above.

    if 'allowed_file' returns False, display the below error message and redirect to 'upload.html' with GET request.
    'Only CSV Files are allowed as Input.'

    if 'allowed_file' returns True, save the file in 'inputdata' folder and
    validate the uploaded csv file using 'validate_input_dataset' defined above.

    if 'validate_input_dataset' returns 'False', remove the file from 'inputdata' folder,
    redirect to 'upload.html' with GET request and respective error message.

    if 'validate_input_dataset' returns 'True', create a 'File' object and save it in database, and
    render 'display_files' template with template varaible 'success_file', set to filename of uploaded file.

    '''
    if request.method =='GET':
      return render_template('upload.html')

    if request.method =='POST':
      f = request.files['file']
      f.save(os.path.join(os.path.join(os.path.join(os.getcwd(), 'spamfilter/inputdata/')),secure_filename(f.filename)))
      #os.path.join(os.getcwd(), 'spamfilter/inputdata/')
      uploadedFile = f.filename
      print("In file_upload() ", os.getcwd(), uploadedFile)
      if allowed_file(uploadedFile, extensions=['.csv']) == True:
        print("In file_upload() ", os.getcwd(), uploadedFile)
        input_dataset_path = os.path.join(os.path.join(os.path.join(os.getcwd(), 'spamfilter/inputdata/')),uploadedFile)
        print("In file_upload() input_dataset_path ", input_dataset_path)
        if validate_input_dataset(input_dataset_path) == True:
          FileObject = File(name=uploadedFile, filepath=input_dataset_path)
          db.session.add(FileObject)  # Adds FileObject record to database
          db.session.commit()  # Commits all change
          return display_files(success_file=uploadedFile)
        else:
          errorMsg = 'invalid dataset'
          print("In file_upload() : invalid dataset ", input_dataset_path)
          os.remove(input_dataset_path)
          #os.system('del inputdatafilepath')
          flash('Only CSV Files are allowed as Input')
          return redirect('upload.html')


      else:
        errorMsg = 'Only CSV Files are allowed as Input.'
        print("In file_upload() ", os.getcwd(), errorMsg)
        flash('Only CSV Files are allowed as Input')
        flash(b'Only CSV Files are allowed as Input')
        return redirect('upload.html')
        #TBC





def validate_input_text(intext):
    '''
    Validate the following details of input email text, provided for prediction.

    1. If the input email text contains more than one mail, they must be separated by atleast one blank line.

    2. Every input email must start with 'Subject:' pattern.

    Return False if any of the two validations fail.

    If all validations pass, Return an Ordered Dicitionary, whose keys are first 30 characters of each
    input email and values being the complete email text.
    '''
    input_list = intext.split('\n')

    print(input_list)
    dic = OrderedDict()
    if len(input_list) == 1 :
      if input_list[0].count("Subject:") > 1 :
        return False
    for email in input_list :
      if email.startswith("Subject:") :
        dic[email[:30]] = email
        pass
      else :
        return False

    return dic


@spam_api.route('/models/<success_model>/')
@spam_api.route('/models/')
def display_models(success_model=None):

    '''
    Obtain the filenames of all machine learning models present in 'mlmodels' folder and
    pass it to template variable 'files'.

    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.

    Consider only the model and not the word_features.pk files.

    Renders 'modelslist.html' template with values  of varaibles 'files' and 'model_name'.
    'model_name' is set to value of 'success_model' argument.

    if 'success_model value is passed, corresponding model file name is highlighted.
    '''
    print("In display_models 1", os.getcwd())
    ListOfFiles=[]
    entries = os.listdir(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'))
    for entry in entries:
        if entry.find('word_features') == -1:
            ListOfFiles.append(entry)
    print("In display_models 2", ListOfFiles, success_model)
    return render_template('modelslist.html', files=ListOfFiles, model_name=success_model)


def isFloat(value):
    '''
    Return True if <value> is a float, else return False
    '''
    return isinstance(value, float)


def isInt(value):
    '''
    Return True if <value> is an integer, else return False
    '''
    return isinstance(value, int)

@spam_api.route('/train/', methods=['GET', 'POST'])
def train_dataset():

    '''
    If request is of GET method, render 'train.html' template with template variable 'train_files',
    set to list if csv files present in 'inputdata' folder.

    If request is of POST method, capture values associated with
    'train_file', 'train_size', 'random_state', and 'shuffle'

    if no 'train_file' is selected, render the same page with GET Request and below error message.
    'No CSV file is selected'

    if 'train_size' is None, render the same page with GET Request and below error message.
    'No value provided for size of training data set.'

    if 'train_size' value is not float, render the same page with GET Request and below error message.
    'Training Data Set Size must be a float.

    if 'train_size' value is not in between 0.0 and 1.0, render the same page with GET Request and below error message.
    'Training Data Set Size Value must be in between 0.0 and 1.0'

    if 'random_state' is None,render the same page with GET Request and below error message.
    'No value provided for random state.''

    if 'random_state' value is not an integer, render the same page with GET Request and below error message.
    'Random State must be an integer.'

    if 'shuffle' is None, render the same page with GET Request and below error message.
    'No option for shuffle is selected.'

    if 'shuffle' is set to 'No' when 'Startify' is set to 'Yes', render the same page with GET Request and below error message.
    'When Shuffle is No, Startify cannot be Yes.'

    If all input values are valid, build the model using submitted paramters and methods defined in
    'spamclassifier.py' and save the model and model word features file in 'mlmodels' folder.

    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.

    Finally render, 'display_models' template with value of template varaible 'success_model'
    set to name of model generated, ie. 'sample.pk'
    '''
    #print("In train_dataset() ", dict(request.form),request.method, dict(request.files, ))
    #print("In train_dataset() ", dict(request.form),request.method, request.form.get('train_file'),request.form.get('random_state'))
    if request.method =='GET':
      print("In train_dataset() GET ", os.getcwd())
      Input_directory = os.path.join(os.getcwd(), 'spamfilter/inputdata/')
      ListOfFiles = os.listdir(Input_directory)
      ListOfCsvFiles = []
      for file in ListOfFiles:
        name, ext = os.path.splitext(file)
        if ext == '.csv':
          ListOfCsvFiles.append(file)
      print("In train_dataset() GET ", Input_directory,ListOfCsvFiles)
      return render_template('train.html', train_files=ListOfCsvFiles)

    else:
      train_file = request.form.get('train_file')
      train_size = request.form.get('train_size')
      random_state = request.form.get('random_state')
      shuffle = request.form.get('shuffle')
      startify = request.form.get('stratify')

      print("In POST ",train_file, type(train_size), random_state, shuffle, startify)
      if train_file == None:
        flash('No CSV file is selected')
        print("train_file ",train_file)
        return redirect(request.url)

      if train_size == None:
        flash('No value provided for size of training data set.')
        print("train_size 1 ",train_size)
        return redirect(request.url)
      elif train_size.isalpha() == True:
        flash('Training Data Set Size must be a float.')
        print("train_size 1.1 ",train_size)
        return redirect(request.url)
      else:
        if isFloat(float(train_size)) == False :
          flash('Training Data Set Size must be a float.')
          print("train_size 2",train_size)
          return redirect(request.url)
        elif (float(train_size) < 0.0 or float(train_size) > 1.0):
          print(train_size)
          flash('Training Data Set Size Value must be in between 0.0 and 1.0')
          print("train_size 3 ",train_size)
          return redirect(request.url)
          print('Random State',random_state)
        else:
          if random_state == None:
            flash('No value provided for random state.')
            print("random_state 1 ",random_state)
            return redirect(request.url)
          elif isInt(int(random_state)) == False:
            flash('Random State must be an integer.')
            print("random_state 2",random_state)
            return redirect(request.url)
          elif shuffle == None:
            flash('No option for shuffle is selected.')
            print("shuffle 1",shuffle)
            return redirect(request.url)
          elif shuffle == 'No' and startify == 'Yes':
            flash('When Shuffle is No, Startify cannot be Yes.')
            print("shuffle 2",shuffle)
            return redirect(request.url)
          else:
            print("ALL WELL")
            data = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'spamfilter/inputdata/'),train_file))
            train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                              data["spam"].values,
                              train_size = float(train_size),
                              random_state = int(random_state))

            classifier = spamclassifier.SpamClassifier()
            classifier_model, model_word_features = classifier.train(train_X, train_Y)
            model_name = train_file.split('.')[0]
            model_file = os.path.join(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'), model_name+'.pk')
            model_word_features_file = os.path.join(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'),model_name +'_word_features.pk')
      #model_word_features_name = 'sample_emails_word_features.pk'
            with open(model_file, 'wb') as model_fp:
              pickle.dump(classifier_model, model_fp)
            with open(model_word_features_file, 'wb') as model_fp:
              pickle.dump(model_word_features, model_fp)

            return display_models(success_model=model_name+'.pk')


@spam_api.route('/results/')
def display_results():
    '''
    Read the contents of 'predictions.json' and pass those values to 'predictions' template varaible

    Render 'displayresults.html' with value of 'predictions' template variable.
    '''
    ListOfResults = []
    with open(os.path.join(os.getcwd(),'predictions.json')) as f:
        datastore = json.load(f)
        ListOfResults = [(k, v) for k, v in datastore.items()]
        print("display_results()", ListOfResults)
    return render_template('displayresults.html', predictions=ListOfResults)


@spam_api.route('/predict/', methods=['GET', "POST"])
def predict():
    '''
    If request is of GET method, render 'emailsubmit.html' template with value of template
    variable 'form' set to instance of 'InputForm'(defined in 'forms.py').
    Set the 'inputmodel' choices to names of models (in 'mlmodels' folder), with out extension i.e .pk

    If request is of POST method, perform the below checks

    1. If input emails is not provided either in text area or as a '.txt' file, render the same page with GET Request and below error message.
    'No Input: Provide a Single or Multiple Emails as Input.'

    2. If input is provided both in text area and as a file, render the same page with GET Request and below error message.
    'Two Inputs Provided: Provide Only One Input.'

    3. In case if input is provided as a '.txt' file, save the uploaded file into 'inputdata' folder and read the
     contents of file into a variable 'input_txt'

    4. If input provided in text area, capture the contents in the same variable 'input_txt'.

    5. validate 'input_txt', using 'validate_input_text' function defined above.

    6. If 'validate_input_text' returns False, render the same page with GET Request and below error message.
    'Unexpected Format : Input Text is not in Specified Format.'


    7. If 'validate_input_text' returns a Ordered dictionary, choose a model and perform prediction of each input email using 'predict' method defined in 'spamclassifier.py'

    8. If no input model is choosen, render the same page with GET Request and below error message.
    'Please Choose a single Model'

    9. Convert the ordered dictionary of predictions, with 0 and 1 values, to another ordered dictionary with values 'NOT SPAM' and 'SPAM' respectively.

    10. Save thus obtained predictions ordered dictionary into 'predictions.json' file.

    11. Render the template 'display_results'

    '''
    ListOfFiles=[]
    entries = os.listdir(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'))
    for entry in entries:
        if entry.find('word_features') == -1:
            name, ext = os.path.splitext(entry)
            if ext == '.pk':
                ListOfFiles.append((name,name))

    if request.method == 'GET':
        newForm = InputForm()
        newForm.inputmodel.choices = ListOfFiles
        print("predict get : ", ListOfFiles, newForm.inputemail, newForm.inputfile)
        print("predict get : ", newForm.inputmodel.choices)
        return render_template('emailsubmit.html',form=newForm)
    else:
        print("predict 1", dict(request.form))
        inputEmail = request.form.get('inputemail')
        inputfile = request.files['inputfile']
        inputmodel = request.form.get('inputmodel')
        print("predict 2",inputEmail, inputmodel)
        if len(inputEmail) == 0 and len(inputfile.filename) == 0:
            flash('No Input: Provide a Single or Multiple Emails as Input.')
            return redirect(url_for('.predict'))
        elif len(inputEmail) != 0 and len(inputfile.filename)!=0:
            flash('Two Inputs Provided: Provide Only One Input.')
            return redirect(url_for('.predict'))

        elif len(inputEmail) == 0 and len(inputfile.filename)!=0:
            inputfile.save(os.path.join(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'),inputfile.filename))
            input_text = inputfile.read()
        elif len(inputEmail) != 0 and len(inputfile.filename)==0:
            input_text = inputEmail


        ret_message = validate_input_text(input_text)
        pred_out = OrderedDict()
        print("predict 3",ret_message)
        if ret_message == False:
            flash('Unexpected Format : Input Text is not in Specified Format.')
            return redirect(url_for('.predict'))

        else:
            if inputmodel == None:
                flash('Please Choose a single Model')
                return redirect(url_for('.predict'))
            else:
                print("predict 4",ret_message)
                classifierclass = spamclassifier.SpamClassifier()
                classifierclass.classifier = pickle.load(open(os.path.join(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'),inputmodel+'.pk'), 'rb'))
                classifierclass.word_features = pickle.load(open(os.path.join(os.path.join(os.getcwd(), 'spamfilter/mlmodels/'),inputmodel+'_word_features.pk'), 'rb'))
                pred_out = classifierclass.predict(ret_message)
                print("predict 5", pred_out)

                for man in pred_out:
                    if pred_out[man] == 0:
                        pred_out[man] = 'SPAM'
                    else:
                        pred_out[man] = 'NOT SPAM'

                with open('predictions.json', 'w') as fp:
                    json.dump(pred_out, fp)

                return display_results()

