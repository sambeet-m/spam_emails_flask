from wtforms import Form, TextAreaField, SubmitField, FileField, validators, RadioField


class InputForm(Form):
    '''
    Include 4 fields : 1. inputemail - a Text Area Field
                       2. inputfile - a File Field
                       3. inputmodel - a Radio Button
                       4. submit - a Submit Button
    '''
    inputemail = TextAreaField('inputemail')
    inputfile  = FileField('inputfile')
    inputmodel = RadioField('inputmodel')
    submit 	   = SubmitField('submit')
