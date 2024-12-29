import csv
import shutil
import cv2
import os
from flask import Flask, request, render_template, session, redirect, g, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# ======== Flask App ========
app = Flask(__name__)
app.secret_key = os.urandom(24)


# ======= Flask Error Handler =======
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('error.html')


# ======= Flask Assign Admin ========
@app.before_request
def before_request():
    g.user = None

    if 'admin' in session:
        g.user = session['admin']


# ======== Current Date & Time =========
datetoday = date.today().strftime("%d-%m-%Y")
datetoday2 = date.today().strftime("%d %B %Y")

# ======== Capture Video ==========
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('UserList'):
    os.makedirs('UserList')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/{datetoday}.csv', 'w') as f:
        f.write('Name,ID,Section,Time')
if 'Registered.csv' not in os.listdir('UserList'):
    with open('UserList/Registered.csv', 'w') as f:
        f.write('Name,ID,Section')
if 'Unregistered.csv' not in os.listdir('UserList'):
    with open('UserList/Unregistered.csv', 'w') as f:
        f.write('Name,ID,Section')


# ======= Remove Empty Rows From Excel Sheet =======
# def remove_empty_cells():
#     dfr = pd.read_csv('UserList/Registered.csv')
#     dfu = pd.read_csv('UserList/Unregistered.csv')

#     dfr.dropna(inplace=True)
#     dfr.to_csv('UserList/Registered.csv', index=False)
#     dfu.dropna(inplace=True)
#     dfu.to_csv('UserList/Unregistered.csv', index=False)

#     for file in os.listdir('Attendance'):
#         csv = pd.read_csv(f'Attendance/{file}')

#         csv.dropna(inplace=True)
#         csv.to_csv(f'Attendance/{file}', index=False)


# ======= Remove Empty Rows From Excel Sheet =======
def remove_empty_cells():
    # Process Registered.csv
    dfr = pd.read_csv('UserList/Registered.csv')
    dfr.dropna(how='all', inplace=True)  # Remove rows only if all cells are NaN
    dfr.to_csv('UserList/Registered.csv', index=False)

    # Process Unregistered.csv
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfu.dropna(how='all', inplace=True)  # Remove rows only if all cells are NaN
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    # Process files in the Attendance folder
    for file in os.listdir('Attendance'):
        if file.endswith('.csv'):  # Ensure only CSV files are processed
            csv = pd.read_csv(f'Attendance/{file}')
            csv.dropna(how='all', inplace=True)  # Remove rows only if all cells are NaN
            csv.to_csv(f'Attendance/{file}', index=False)



# ======= Total Registered Users ========
def totalreg():
    return len(os.listdir('static/faces'))


# ======= Get Face From Image =========
def extract_faces(img):
    if img is None:
        return []
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray_img, 1.3, 5)
        return face_points
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

# ======= Identify Face Using ML ========
def identify_face(face_array):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        prediction = model.predict(face_array)
        
        # Check if the prediction is valid and not empty
        if prediction:
            return prediction
        else:
            return ["Not registered"]  # Return a custom message if no valid prediction
    except Exception as e:
        print(f"Error in face identification: {e}")
        return ["Not registered"]

    
# ======= Train Model Using Available Faces ========
def train_model():
    # Check if model exists and remove it
    if 'face_recognition_model.pkl' in os.listdir('static'):
        os.remove('static/face_recognition_model.pkl')

    # Check if there are faces in the directory
    if len(os.listdir('static/faces')) == 0:
        print("No faces found for training.")
        return

    # Load the registered users list from the CSV
    registered_users = pd.read_csv('UserList/Registered.csv')
    
    faces = []
    labels = []

    # Iterate over each registered user and train only if they have a section value
    for _, row in registered_users.iterrows():
        user_name = row['Name']
        user_id = row['ID']
        user_section = row['Section']

        if user_section is not None:  # Only train if the section is not None
            # Construct the folder path based on name, ID, and section
            user_folder_path = f'static/faces/{user_name}${user_id}${user_section}'

            if os.path.exists(user_folder_path):  # Ensure the user's folder exists
                # Iterate over each image in the user's folder
                for img_name in os.listdir(user_folder_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Filter only image files
                        img_path = os.path.join(user_folder_path, img_name)
                        img = cv2.imread(img_path)

                        # Ensure the image was loaded correctly
                        if img is not None:
                            resized_face = cv2.resize(img, (50, 50))  # Resize the face image
                            faces.append(resized_face.ravel())  # Flatten the image for training
                            labels.append(f"{user_name}${user_id}${user_section}")  # Create label with Name, ID, and Section

    # If no faces were found, return early
    if len(faces) == 0:
        print("No valid face images found for training.")
        return

    # Convert faces list to numpy array
    faces = np.array(faces)
    
    # Set number of neighbors for KNN based on the number of samples
    n_samples = len(faces)
    n_neighbors = max(1, min(5, n_samples))  # Ensure n_neighbors is at least 1

    # Train the KNeighborsClassifier model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(faces, labels)

    # Save the trained model to a file
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    print("Model trained and saved successfully.")

# ======= Remove Attendance of Deleted User ======
def remAttendance():
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    for file in os.listdir('Attendance'):
        df = pd.read_csv(f'Attendance/{file}')
        df.reset_index()
        csv_file = csv.reader(open(f'Attendance/' + file, "r"), delimiter=",")

        skip_header = True
        i = 0
        for row in csv_file:
            if not row:
                continue

            if skip_header:
                skip_header = False
                continue

            if str(row[1]) not in list(map(str, dfu['ID'])) and str(row[1]) not in list(map(str, dfr['ID'])):
                df.drop(df.index[i], inplace=True)
                df.to_csv(f'Attendance/{file}', index=False)

            i += 1

    remove_empty_cells()


# ======== Get Info From Attendance File =========
def extract_attendance():
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')
    df = pd.read_csv(f'Attendance/{datetoday}.csv')

    names = df['Name']
    rolls = df['ID']
    sec = df['Section']
    times = df['Time']
    dates = f'{datetoday}'

    reg = []
    roll = list(rolls)
    for i in range(len(df)):
        if str(roll[i]) in list(map(str, dfu['ID'])):
            reg.append("Unregistered")
        elif str(roll[i]) in list(map(str, dfr['ID'])):
            reg.append("Registered")

    l = len(df)
    return names, rolls, sec, times, dates, reg, l


# ======== Save Attendance =========
def add_attendance(name):
    username = name.split('$')[0]
    userid = name.split('$')[1]  # Ensure this is treated as a string
    usersection = name.split('$')[2]
    current_time = datetime.now().strftime("%I:%M %p")
    
    attendance_file = f'Attendance/{datetoday}.csv'

    if not os.path.exists(attendance_file):
        # Create file with correct headers if not present
        with open(attendance_file, 'w') as f:
            f.write('Name,ID,Section,Time\n')

    df = pd.read_csv(attendance_file)
    
    # Ensure the comparison treats both the ID as strings
    if str(userid) in map(str, df['ID']):
        last_attendance_time = df[df['ID'] == str(userid)].iloc[-1]['Time']
        start_time = datetime.strptime(last_attendance_time, "%I:%M %p")
        end_time = datetime.strptime(current_time, "%I:%M %p")
        delta = (end_time - start_time).total_seconds() / 60

        if delta < 60:
            return  # Skip attendance if less than 1 hour

    # Create new row with correct data and calculate the new index
    new_row = pd.DataFrame({'Name': [username], 'ID': [userid], 'Section': [usersection], 'Time': [current_time]})
    
    # Concatenate the new row with the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(attendance_file, index=False)

# ======= Flask Home Page =========
@app.route('/')
def home():
    if g.user:
        return render_template('home.html', admin=True, mess='Logged in as Administrator', user=session['admin'])

    return render_template('home.html', admin=False, datetoday2=datetoday2)


# ======== Flask Take Attendance ==========
@app.route('/attendance')
def attendance():
    if f'{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/{datetoday}.csv', 'w') as f:
            f.write('Name,ID,Section,Time')

    remove_empty_cells()
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)


@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    # Check if faces directory is empty
    if len(os.listdir('static/faces')) == 0:
        return render_template('attendance.html', datetoday2=datetoday2, 
                               mess='Database is empty! Register yourself first.')

    # Check if the face recognition model exists, else train it
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        train_model()

    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Camera is not available.")
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    ret = True
    recent_users = {}  # Dictionary to track last attendance time for users
    
    # Load the registered users list
    registered_users = pd.read_csv('UserList/Registered.csv')
    registered_ids = registered_users['ID'].tolist()  # List of registered user IDs

    while ret:
        ret, frame = cap.read()
        
        if frame is None:
            continue
        
        # Detect faces in the frame
        faces = extract_faces(frame)
        
        if len(faces) > 0:  # Check if any faces were detected
            (x, y, w, h) = faces[0]  # Use the first detected face
            try:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_result = identify_face(face.reshape(1, -1))
                
                if len(identified_result) > 0:
                    identified_person = identified_result[0]
                    if identified_person == "Not registered":
                        # Show "Not registered" message
                        cv2.putText(frame, "User not registered. Attendance not marked.", (30, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        print("User not registered.")
                    else:
                        person_details = identified_person.split('$')

                        if len(person_details) == 3:
                            identified_person_name = person_details[0]
                            identified_person_id = person_details[1]
                            identified_person_section = person_details[2]
                            
                            # Check if the identified user is in the registered users list
                            if identified_person_id in registered_ids:
                                current_time = datetime.now()

                                if (identified_person_id not in recent_users or 
                                    (current_time - recent_users[identified_person_id]).total_seconds() > 3600):
                                    # Mark attendance if not marked in the past hour
                                    add_attendance(identified_person)
                                    recent_users[identified_person_id] = current_time
                                    message = f"Attendance marked for {identified_person_name}."
                                else:
                                    message = f"Attendance already marked recently."

                                # Dynamically adjust the position and font size for display
                                cv2.putText(frame, f'Name: {identified_person_name}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                                cv2.putText(frame, f'Status: {message}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                            else:
                                # If the person is not registered, show an error message
                                message = "User not registered. Attendance not marked."
                                cv2.putText(frame, message, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                print(f"{identified_person_name} is not registered.")
                        else:
                            print(f"Error: Invalid identification format: {identified_person}")
                else:
                    print("No valid result from face recognition.")
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        cv2.putText(frame, 'Press Esc to close', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 127, 255), 2, cv2.LINE_AA)
        
        # Create the window and display the frame
        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
        cv2.imshow('Attendance', frame)

        # Break the loop when the ESC key is pressed
        if cv2.waitKey(50) == 27:  # Wait for 50ms, break on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

    # Now use pd.concat instead of append to update the attendance
    names, rolls, sec, times, dates, reg, l = extract_attendance()

    # Render the attendance page with updated data
    return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)


# ========== Flask Add New User ============
@app.route('/adduser')
def adduser():
    return render_template('adduser.html')


@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newusersection = request.form['newusersection']

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return render_template('adduser.html', mess='Camera not available.')

    userimagefolder = f'static/faces/{newusername}${newuserid}${newusersection}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    if str(newuserid) in map(str, dfr['ID']):
        return render_template('adduser.html', mess='You are already a registered user.')
    elif str(newuserid) in map(str, dfu['ID']):
        return render_template('adduser.html', mess='You are already in the unregistered list.')

    photos_captured = 0
    max_photos = 10
    exited_early = False

    while photos_captured < max_photos:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)  # Detect faces
        for (x, y, w, h) in faces[:1]:  # Process only the first detected face
            cropped_face = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(cropped_face, (150, 150))  # Resize to consistent dimensions
            photo_name = f'{newusername}_{photos_captured + 1}.jpg'
            cv2.imwrite(os.path.join(userimagefolder, photo_name), resized_face)
            photos_captured += 1

            # Break early if maximum photos are captured
            if photos_captured >= max_photos:
                break

        # Display progress
        progress_frame = frame.copy()
        cv2.putText(progress_frame, f'Capturing Photos: {photos_captured}/{max_photos}', (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.namedWindow('Collecting Face Data', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Collecting Face Data', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Collecting Face Data', progress_frame)

        if cv2.waitKey(1) == 27:  # Exit if 'Esc' is pressed
            exited_early = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if photos_captured < max_photos:
        # Cleanup if photos were not captured or exited early
        shutil.rmtree(userimagefolder)
        if not exited_early:
            return render_template('adduser.html', mess='Failed to Capture Photos.')
        else:
            return render_template('adduser.html', mess='Photo capture process exited.')

    # Add to CSV only if images were successfully captured
    with open('UserList/Unregistered.csv', 'a') as f:
        f.write(f'\n{newusername},{newuserid},{newusersection}')

    # train_model()
    return render_template('adduser.html', mess='Waiting for admin approval. You are listed as Unregistered.')

# ========== Flask Attendance List ============
@app.route('/attendancelist')
def attendancelist():
    if not g.user:
        return render_template('login.html')

    remove_empty_cells()

    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg,
                           l=0)


# ========== Flask Search Attendance by Date ============
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('login.html')

    date = request.form['date']

    year = date.split('-')[0]
    month = date.split('-')[1]
    day = date.split('-')[2]

    if f'{day}-{month}-{year}.csv' not in os.listdir('Attendance'):
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=0,
                               mess="Nothing Found!")
    else:
        names = []
        rolls = []
        sec = []
        times = []
        dates = []
        reg = []
        l = 0

        skip_header = True
        csv_file = csv.reader(open(f'Attendance/{day}-{month}-{year}.csv', "r"), delimiter=",")
        dfu = pd.read_csv('UserList/Unregistered.csv')
        dfr = pd.read_csv('UserList/Registered.csv')

        for row in csv_file:
            if skip_header:
                skip_header = False
                continue

            names.append(row[0])
            rolls.append(row[1])
            sec.append(row[2])
            times.append(row[3])
            dates.append(f'{day}-{month}-{year}')

            if str(row[1]) in list(map(str, dfu['ID'])):
                reg.append("Unregistered")
            elif str(row[1]) in list(map(str, dfr['ID'])):
                reg.append("Registered")
            else:
                reg.append("x")

            l += 1

        if l != 0:
            return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                                   reg=reg, l=l,
                                   totalreg=totalreg())
        else:
            return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                                   reg=reg, l=l,
                                   totalreg=totalreg(),
                                   mess="Nothing Found!")


# ========== Flask Search Attendance by ID ============
@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('login.html')

    id = request.form['id']

    names = []
    rolls = []
    sec = []
    times = []
    dates = []
    reg = []
    l = 0

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    for file in os.listdir('Attendance'):
        csv_file = csv.reader(open('Attendance/' + file, "r"), delimiter=",")

        for row in csv_file:
            if row[1] == id:
                names.append(row[0])
                rolls.append(row[1])
                sec.append(row[2])
                times.append(row[3])
                dates.append(file.replace('.csv', ''))

                if str(row[1]) in list(map(str, dfu['ID'])):
                    reg.append("Unregistered")
                elif str(row[1]) in list(map(str, dfr['ID'])):
                    reg.append("Registered")
                else:
                    reg.append("x")

                l += 1

    if l != 0:
        return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=l,
                               mess=f'Total Attendance: {l}')
    else:
        return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=l,
                               mess="Nothing Found!")


# ========== Flask Registered Users ============
@app.route('/registereduserlist')
def registereduserlist():
    if not g.user:
        return render_template('login.html')

    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Unregister a User ============
@app.route('/unregisteruser', methods=['GET', 'POST'])
def unregisteruser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    # Remove any empty cells from the DataFrame
    remove_empty_cells()

    # Load the registered and unregistered users data
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    # Extract the row to be removed from the registered list
    row = dfr.iloc[[idx]]

    # Move the face data to a 'None' section
    shutil.move('static/faces/' + dfr.iloc[idx]['Name'] + '$' + dfr.iloc[idx]['ID'] + '$' + dfr.iloc[idx]['Section'],
                'static/faces/' + dfr.iloc[idx]['Name'] + '$' + dfr.iloc[idx]['ID'] + '$None')

    # Replace the section with 'None'
    row['Section'] = row['Section'].replace(to_replace='.', value='None', regex=True)

    # Use pd.concat() to append the row to the Unregistered.csv
    dfu = pd.concat([dfu, row], ignore_index=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    # Drop the user from the Registered.csv
    dfr.drop(dfr.index[idx], inplace=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    # Remove any empty cells from the DataFrame again
    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    # Read the updated registered users list
    skip_header = True
    with open('UserList/Registered.csv', "r") as file:
        csv_file = csv.reader(file, delimiter=",")
        for row in csv_file:
            if skip_header:
                skip_header = False
                continue

            names.append(row[0])
            rolls.append(row[1])
            sec.append(row[2])
            l += 1

    # Re-train the model after the update
    train_model()

    # Return the updated registered users list
    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")

# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['GET', 'POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    remove_empty_cells()
    dfr = pd.read_csv('UserList/Registered.csv')
    username = dfr.iloc[idx]['Name']
    userid = dfr.iloc[idx]['ID']
    usersec = dfr.iloc[idx]['Section']

    if f'{username}${userid}${usersec}' in os.listdir('static/faces'):
        shutil.rmtree(f'static/faces/{username}${userid}${usersec}')
        train_model()

    dfr.drop(dfr.index[idx], inplace=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    remAttendance()

    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Unregistered Users ============
@app.route('/unregistereduserlist')
def unregistereduserlist():
    if not g.user:
        return render_template('login.html')

    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Unregistered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Unregistered Students: {l}')
    else:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Register a User ============
@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if not g.user:
        return render_template('login.html')

    # Get the index and section from the form
    idx = int(request.form['index'])
    sec = request.form['section'].strip()

    # Remove empty cells
    remove_empty_cells()

    # Read Unregistered and Registered CSV files
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    # Get the row to move from Unregistered to Registered
    row = dfu.iloc[[idx]].copy()

    # Construct file paths for moving
    source_path = os.path.join('static/faces', f"{dfu.iloc[idx]['Name']}${dfu.iloc[idx]['ID']}$None")
    destination_path = os.path.join('static/faces', f"{dfu.iloc[idx]['Name']}${dfu.iloc[idx]['ID']}${sec}")

    # Check if source exists and move the file
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
    else:
        return f"Error: File not found at {source_path}"

    # Remove the previous folder if it exists (after moving)
    previous_folder_path = os.path.join('static/faces', f"{dfu.iloc[idx]['Name']}${dfu.iloc[idx]['ID']}$None")
    if os.path.isdir(previous_folder_path):
        shutil.rmtree(previous_folder_path)  # Remove the previous folder

    # Update the section in the row and add it to Registered.csv
    row['Section'] = sec
    dfr = pd.concat([dfr, row], ignore_index=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    # Remove the row from Unregistered.csv
    dfu.drop(dfu.index[idx], inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    # Clean up any empty cells
    remove_empty_cells()

    # Prepare data for the template
    names, rolls, sec_list = [], [], []
    l = 0

    # Read remaining Unregistered users
    with open('UserList/Unregistered.csv', "r") as file:
        csv_file = csv.reader(file, delimiter=",")
        next(csv_file, None)  # Skip header
        for row in csv_file:
            names.append(row[0])
            rolls.append(row[1])
            sec_list.append(row[2])
            l += 1

    # Render the updated Unregistered User List
    if l > 0:
        message = f'Number of Unregistered Students: {l}'
    else:
        message = "Database is empty!"
    train_model()

    return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec_list, l=l, mess=message)



# ========== Flask Delete a User from Unregistered List ============
@app.route('/deleteunregistereduser', methods=['GET', 'POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    remove_empty_cells()
    dfu = pd.read_csv('UserList/Unregistered.csv')
    username = dfu.iloc[idx]['Name']
    userid = dfu.iloc[idx]['ID']
    usersec = dfu.iloc[idx]['Section']

    print(f'{username}${userid}${usersec}')
    print(os.listdir('static/faces'))
    if f'{username}${userid}${usersec}' in os.listdir('static/faces'):
        shutil.rmtree(f'static/faces/{username}${userid}${usersec}')
        train_model()

    dfu.drop(dfu.index[idx], inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Unregistered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    remAttendance()

    if l != 0:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Unregistered Students: {l}')
    else:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========= Flask Admin Login ============
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '12345':
            session['admin'] = request.form['username']
            return redirect(url_for('home', admin=True, mess='Logged in as Administrator'))
        else:
            return render_template('login.html', mess='Incorrect Username or Password')

    return render_template('login.html')


# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('login.html')


# ======= Main Function =========
if __name__ == '__main__':
    app.run(debug=True)




