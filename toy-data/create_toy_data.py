import pandas as pd
import numpy as np

np.random.seed(4)

app = pd.read_csv('../data/Appointments.csv', low_memory=False)
oq = pd.read_csv('../data/OQ.csv', low_memory=False)
pat_info = pd.read_csv('../data/PatientInformation.csv', low_memory=False)
with open('../data/for_toy_data.txt') as f:
   toy_text = f.readlines()

app['ClientID'] = pd.to_numeric(app['ClientID'], errors='coerce')
oq['ClientID'] = pd.to_numeric(oq['ClientID'], errors='coerce')
pat_info['ClientID'] = pd.to_numeric(pat_info['ClientID'], errors='coerce')

# reference clients for a sample of dates 

def ref_clients():
    app['Date'] = pd.to_datetime(app["Date"])
    app['ClientID'] = pd.to_numeric(app['ClientID'], errors='coerce')
    oq['ClientID'] = pd.to_numeric(oq['ClientID'], errors='coerce')
    pat_info['ClientID'] = pd.to_numeric(pat_info['ClientID'], errors='coerce')
    
    start = pd.to_datetime(toy_text[0])
    end = pd.to_datetime(toy_text[1])
    app.loc[(app['Date'] >= start) & (app['Date'] <= end), "ClientID" ]
    client_list = app.loc[(app['Date'] >= start) & (app['Date'] <= end), "ClientID" ].unique()

    ref_app = pd.DataFrame(columns=['ClientID', 'TherapistID', 'Date'])
    ref_oq = pd.DataFrame(columns=['ClientID', 'AdministrationDate'])
    ref_pat = pd.DataFrame(columns=['ClientID', 'notedate'])

    ref_clients_list = np.random.choice(client_list, size=len(client_ids), replace=False)

    for ref_client in ref_clients_list:
        A = app.loc[app["ClientID"] == ref_client, ref_app.columns]
        ref_app = pd.concat([ref_app, A], axis=1)
        A = oq.loc[oq["ClientID"] == ref_client, ref_oq.columns]
        ref_oq = pd.concat([ref_oq, A], axis=1)
        A = pat_info.loc[pat_info["ClientID"] == ref_client, ref_pat.columns]
        ref_pat = pd.concat([ref_pat, A], axis=1)

    # modify data
    # client id
    replace_client = dict()
    for i, client in enumerate(ref_clients_list):
        replace_client[client] = i + 1000000

    ref_app["ClientID"] = ref_app.replace(replace_client)["ClientID"]
    ref_oq["ClientID"] = ref_oq.replace(replace_client)["ClientID"]
    ref_pat["ClientID"] = ref_pat.replace(replace_client)["ClientID"]

    # therapist id
    replace_th = dict()
    for i, therapist in enumerate(new_app['TherapistID']):
        replace_th[therapist] = i + 100

    new_app['TherapistID'] = new_app.replace(replace_th)['TherapistID']

    # date
    ref_app["Date"] += pd.DateOffset(years=toy_text[2], days=toy_text[3])
    ref_oq["AdministrationDate"] += pd.DateOffset(years=toy_text[2], days=toy_text[3])
    ref_pat['notedate'] += pd.DateOffset(years=toy_text[2], days=toy_text[3])
    return (ref_app, ref_oq, ref_pat)

input_ref = input("Make date-reference data-frames? (Y/n) ")
if input_ref == "Y":
    ref_app, ref_oq, ref_pat = ref_clients()
    ref_app.to_csv('r_appointments.csv', index=False)
    ref_oq.to_csv('r_oq.csv', index=False)
    ref_pat.to_csv('r_patientinfo.csv', index=False)

ref_app = pd.read_csv('r_appointments.csv')
ref_oq = pd.read_csv('r_oq.csv')
ref_pat = pd.read_csv('r_patientinfo.csv')

### Appointments ###

new_app = pd.DataFrame(columns=app.columns)
new_app['ClientID'] = ref_app['ClientID']
new_app['TherapistID'] = ref_app['TherapistID']
new_app['Date'] = ref_app['Date']


def add_type_attend(new_app, app):
    subset_clients = np.random.choice(app['ClientID'].unique(), size=new_app['ClientID'].nunique())
    for sample_c, client in zip(subset_clients, new_app['ClientID'].unique()):
        mask = new_app['ClientID'] == client
        type = app.loc[app["ClientID"] == sample_c, "AppType"].values
        new_app.loc[mask, 'AppType'] = np.random.sample(type, size=mask.sum())
    
    subset_clients = np.random.choice(app['ClientID'].unique(), size=new_app['ClientID'].nunique())
    for sample_c, client in zip(subset_clients, new_app['ClientID'].unique()):
        mask = new_app['ClientID'] == client
        attend = app.loc[app["ClientID"] == sample_c, 'AttendanceDescription'].values
        new_app.loc[mask, 'AttendanceDescription'] = np.random.sample(attend, size=mask.sum())
    
    return new_app


new_app = add_type_attend(new_app, app)
new_app.to_cvs('Appointments.csv', index=False)

### OQ ####

# new_oq = pd.DataFrame(columns=oq.columns)
