import pandas as pd

# Load the dataset
file_name = "Medical Appointments.csv"
df = pd.read_csv(file_name)

# Display basic information and the first few rows
df.info()
print(df.head())

"""
Dataset Overview:

The dataset contains 110,527 entries and 13 columns. Here is a brief overview of the columns:

1.	PatientId: The ID of the patient.
2.	AppointmentID: The ID of the appointment.
3.	Gender: The gender of the patient.
4.	ScheduledDay: The day the appointment was scheduled.
5.	AppointmentDay: The day of the actual appointment.
6.	Age: The age of the patient.
7.	Neighbourhood: The neighborhood of the patient.
8.	Scholarship: Indicates if the patient is enrolled in the welfare program.
9.	Hipertension: Indicates if the patient has hypertension.
10.	Diabetes: Indicates if the patient has diabetes.
11.	Handcap: Indicates if the patient is handicapped.
12.	SMS_received: Indicates if the patient received an SMS reminder.
13.	No-show: Indicates if the patient did not show up for the appointment.
"""

# Demographic Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Gender distribution
gender_distribution = df["Gender"].value_counts()

# Age distribution
age_distribution = df["Age"]

# Plotting Gender Distribution
sns.countplot(data=df, x="Gender", palette="pastel")
plt.title("Distribution of Gender Among Patients")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Plotting Age Distribution
sns.histplot(age_distribution, bins=30, kde=True, color="skyblue")
plt.title("Age Distribution Among Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

"""
Demographic Analysis
Distribution of Gender Among Patients:

The dataset has 71,840 female patients and 38,687 male patients.
Age Distribution Among Patients:

The age of patients ranges from -1 to 115 years. The average age is approximately 37 years, with a standard deviation of about 23 years.
Outliers or Inconsistencies in Age Distribution:

There is an inconsistency with a minimum age of -1, which is not valid. This entry will need to be addressed as an outlier.
"""

# Addressing the Outlier in Age

# Removing the invalid age entry
df = df[df["Age"] >= 0]

# Re-checking the age distribution after removing outliers
age_distribution_cleaned = df["Age"]

# Plotting the cleaned Age Distribution
sns.histplot(age_distribution_cleaned, bins=30, kde=True, color="skyblue")
plt.title("Cleaned Age Distribution Among Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Appointment Trends

# Converting ScheduledDay and AppointmentDay to datetime format
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

# Extracting the day of the week and month for analysis
df["ScheduledDayOfWeek"] = df["ScheduledDay"].dt.day_name()
df["AppointmentDayOfWeek"] = df["AppointmentDay"].dt.day_name()
df["ScheduledMonth"] = df["ScheduledDay"].dt.month
df["AppointmentMonth"] = df["AppointmentDay"].dt.month

# Plotting the distribution of appointments over different days of the week
sns.countplot(
    data=df,
    x="AppointmentDayOfWeek",
    palette="pastel",
    order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
)
plt.title("Distribution of Appointments Over Different Days of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Count")
plt.show()

# Plotting the distribution of appointments over different months
sns.countplot(data=df, x="AppointmentMonth", palette="pastel")
plt.title("Distribution of Appointments Over Different Months")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

"""
Appointment Trends

Distribution of Appointments Over Different Days of the Week:
Most appointments are scheduled from Monday to Friday, with fewer appointments on Saturdays. There are no appointments on Sundays.

Distribution of Appointments Over Different Months:
Appointments are mostly scheduled in the month of May.
"""

# Patient Characteristics

# Calculating proportions of patients with hypertension, diabetes, or handicaps
hypertension_proportion = df["Hipertension"].mean()
diabetes_proportion = df["Diabetes"].mean()
handicap_proportion = df["Handcap"].mean()

# Plotting the proportions
conditions = ["Hypertension", "Diabetes", "Handicap"]
proportions = [hypertension_proportion, diabetes_proportion, handicap_proportion]

sns.barplot(x=conditions, y=proportions, palette="pastel")
plt.title("Proportion of Patients with Hypertension, Diabetes, or Handicaps")
plt.xlabel("Condition")
plt.ylabel("Proportion")
plt.ylim(0, 0.2)
plt.show()

# Checking correlations between conditions
correlations = df[["Hipertension", "Diabetes", "Handcap"]].corr()
print(correlations)

# Effect of Reminder SMS

# Proportion of patients who received SMS reminders
sms_received_proportion = df["SMS_received"].mean()

# Show-up rates based on SMS reminders
show_up_rate_with_sms = df[df["SMS_received"] == 1]["No-show"].value_counts(
    normalize=True
)["No"]
show_up_rate_without_sms = df[df["SMS_received"] == 0]["No-show"].value_counts(
    normalize=True
)["No"]

# Plotting the show-up rates
labels = ["Received SMS", "Did Not Receive SMS"]
show_up_rates = [show_up_rate_with_sms, show_up_rate_without_sms]

sns.barplot(x=labels, y=show_up_rates, palette="pastel")
plt.title("Show-Up Rates Based on SMS Reminders")
plt.xlabel("SMS Reminder")
plt.ylabel("Show-Up Rate")
plt.ylim(0, 1)
plt.show()

# Interestingly, the show-up rate for patients who received the SMS is lower than those who did not receive the SMS.

# Financial Aid and Attendance

# Number of patients who received scholarships
scholarship_count = df["Scholarship"].sum()

# Attendance rates based on scholarships
attendance_rate_with_scholarship = df[df["Scholarship"] == 1]["No-show"].value_counts(
    normalize=True
)["No"]
attendance_rate_without_scholarship = df[df["Scholarship"] == 0][
    "No-show"
].value_counts(normalize=True)["No"]

# Plotting the attendance rates
labels = ["Received Scholarship", "Did Not Receive Scholarship"]
attendance_rates = [
    attendance_rate_with_scholarship,
    attendance_rate_without_scholarship,
]

sns.barplot(x=labels, y=attendance_rates, palette="pastel")
plt.title("Attendance Rates Based on Scholarships")
plt.xlabel("Scholarship")
plt.ylabel("Attendance Rate")
plt.ylim(0, 1)
plt.show()

# Neighborhood Analysis

# Neighborhoods with the highest number of appointments
neighborhood_appointments = df["Neighbourhood"].value_counts().head(10)

# Plotting the neighborhoods with the highest number of appointments
sns.barplot(
    x=neighborhood_appointments.index,
    y=neighborhood_appointments.values,
    palette="pastel",
)
plt.title("Top 10 Neighborhoods with the Highest Number of Appointments")
plt.xlabel("Neighborhood")
plt.ylabel("Number of Appointments")
plt.xticks(rotation=45)
plt.show()

# Attendance rate based on neighborhood
neighborhood_attendance = (
    df.groupby("Neighbourhood")["No-show"].value_counts(normalize=True).unstack()["No"]
)

# Plotting attendance rates by neighborhood
neighborhood_attendance = neighborhood_attendance.sort_values(ascending=False).head(10)

sns.barplot(
    x=neighborhood_attendance.index, y=neighborhood_attendance.values, palette="pastel"
)
plt.title("Top 10 Neighborhoods by Attendance Rate")
plt.xlabel("Neighborhood")
plt.ylabel("Attendance Rate")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()

# No-Show Analysis

# Overall no-show rate
no_show_rate = df["No-show"].value_counts(normalize=True)["Yes"]

# No-show rate by gender
no_show_rate_by_gender = (
    df.groupby("Gender")["No-show"].value_counts(normalize=True).unstack()["Yes"]
)

# No-show rate by age group
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 18, 35, 55, 75, 100],
    labels=["0-18", "19-35", "36-55", "56-75", "76-100"],
)
no_show_rate_by_age_group = (
    df.groupby("AgeGroup")["No-show"].value_counts(normalize=True).unstack()["Yes"]
)

# Plotting the overall no-show rate
sns.barplot(x=["Overall"], y=[no_show_rate], palette="pastel")
plt.title("Overall No-Show Rate")
plt.xlabel("Category")
plt.ylabel("No-Show Rate")
plt.ylim(0, 1)
plt.show()

# Plotting no-show rate by gender
sns.barplot(
    x=no_show_rate_by_gender.index, y=no_show_rate_by_gender.values, palette="pastel"
)
plt.title("No-Show Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("No-Show Rate")
plt.ylim(0, 1)
plt.show()

# Plotting no-show rate by age group
sns.barplot(
    x=no_show_rate_by_age_group.index,
    y=no_show_rate_by_age_group.values,
    palette="pastel",
)
plt.title("No-Show Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("No-Show Rate")
plt.ylim(0, 1)
plt.show()
