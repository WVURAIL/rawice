import dateutil, pytz, datetime
import numpy as np
import raw_acq_diagnostics as rad
import click
import smtplib
import os
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate


# Set email username and app password
username = None
app_password = None

def send_email(subject, body, sender, recipients, password, files=None):
    msg = MIMEMultipart()
    
    msg.attach(MIMEText(body))
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    
    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    print("Sending email from {} to {}".format(sender, recipients))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Email sent!")


# bad_inputs : list of int
#    An Nx3 list specifying bad inputs to avoid. Each N entry is a coordinate: [crate, slot, input]
BAD_INPUTS = [
    [0, 0, 1],
    [0, 0, 3],
    [0, 0, 6],
    [0, 1, 6],
    [0, 1, 14],
    [0, 2, 6],
    [0, 2, 9],
    [0, 3, 0],
    [0, 3, 2],
    [0, 3, 6],
    [0, 3, 15],
    [0, 4, 4],
    [0, 4, 5],
    [0, 4, 10],
    [0, 5, 3],
    [0, 5, 4],
    [0, 5, 7],
    [0, 5, 9],
    [0, 6, 4],
    [0, 7, 6],
    [0, 8, 2],
    [0, 8, 6],
    [0, 8, 15],
    [0, 9, 5],
    [0, 9, 11],
    [0, 10, 4],
    [0, 10, 8],
    [0, 10, 10],
    [0, 10, 11],
    [0, 10, 14],
    [0, 12, 4],
    [0, 13, 4],
    [0, 14, 3],
    [0, 14, 4],
    [0, 14, 7],
    [0, 14, 10],
    [0, 15, 0],
    [0, 15, 4],
    [0, 15, 5],
    [0, 15, 9],
    [0, 15, 12],
    [0, 15, 15],
] # as of Dec 12, 2023 for GBO


# Define the main CLI group
@click.group()
def cli():
    """
    CHIME/FRB Raw Acquisition Analysis Command Line Interface
    """
    pass

@click.command(
    "plot-summed-spectrum",
    help="Plot a dynamic spectrum summed over all inputs for the given time period.",
)
@click.option(
    "--start-time",
    type=click.STRING,
    default="",
    help="The start time in ET encompassing the data that should be plotted (format YYYY-MM-DD HH:mm:ss, defaults to the last 24 hours).",
)
@click.option(
    "--end-time",
    type=click.STRING,
    default="",
    help="The end time in ET encompassing the data that should be plotted (format YYYY-MM-DD HH:mm:ss, defaults to the last 24 hours).",
)
@click.option(
    "--mask-rfi",
    type=click.BOOL,
    default=True,
    help="Indicates if the three most significant persistent RFI channels will be masked.",
)
@click.option(
    "--mask-sun",
    type=click.BOOL,
    default=True,
    help="Indicates if solar transit will be masked (recommended especially in the Summer).",
)
@click.option(
    "--ds-time-factor",
    type=click.INT,
    default=3,
    help="An integer indicating the factor to reduce the number of time samples by.",
)
@click.option(
    "--ds-freq-factor",
    type=click.INT,
    default=1,
    help="An integer indicating the factor to reduce the number of spectral subbands by.",
)
@click.option(
    "--site",
    type=click.Choice(['gbo', 'chime', 'kko', 'pco', 'hco']),
    default="gbo",
    help="A string indicating the site that the data is being plotted for. Should be one of the following: ['gbo', 'chime', 'kko', 'pco', 'hco']",
)
@click.option(
    "--save-plot",
    type=click.BOOL,
    required=False,
    default=True,
    help="Whether to save the plot to the directory indicated by plot_dir (current directory by default)",
)
@click.option(
    "--plot-dir",
    type=click.Path(exists=False, readable=True, resolve_path=True),
    required=False,
    default='./',
    help="The directory that the plot will be saved to if save_plot=True (current directory by default)",
)
@click.option(
    "--email", "-e",
    type=click.STRING,
    default=[],
    multiple=True,
    help="If provided, script will email the plot to the given email address. Multiple email addresses can be provided.",
)

def plot_summed_spectrum(
        start_time,
        end_time,
        mask_rfi,
        mask_sun,
        ds_time_factor,
        ds_freq_factor,
        site,
        save_plot,
        plot_dir,
        email,
):
    est = pytz.timezone('US/Eastern')
    utc = pytz.utc
    
    # If start_time and end_time are not provided, select the last 24 hours
    if start_time == "" or end_time == "":
        end_time_utc = (datetime.datetime.utcnow() - datetime.timedelta(minutes=5)).astimezone(utc)
        start_time_utc = (end_time_utc - datetime.timedelta(hours=24)).astimezone(utc)
    else:
        # Else, check that start_time and end_time are formatted properly
        try:
            start_time_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_time_dt = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

            # Turn the strings into timezone-aware UTC datetimes
            start_time_et = est.localize(start_time_dt)
            end_time_et = est.localize(end_time_dt)
            start_time_utc = start_time_et.astimezone(utc)
            end_time_utc = end_time_et.astimezone(utc)
        except Exception as e:
            print("Exception encountered: {}".format(e))
            exit()

    dates = np.array([
        start_time_utc,
        end_time_utc,
    ])

    # If plot should not be saved, but an email is provided,
    # just save it temporarily in the current directory so that it can be emailed
    if not save_plot and len(email) != 0:
        plot_dir = './'
    
    raw_acq = rad.RawAcq(
        dates=dates,
        plot_dir=plot_dir,
    )

    raw_acq.plot_total_dynamic_spectrum(
        mask_rfi=mask_rfi,
        mask_sun=mask_sun,
        ds_time_factor=ds_time_factor,
        ds_freq_factor=ds_freq_factor,
        site=site,
        figsize=(7,7),
        save_plot=True,
        bad_inputs=BAD_INPUTS,
    )

    plot_name = "{}/dynamic_spectrum_{}_{}.pdf".format(
        raw_acq.plot_dir,
        raw_acq.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        raw_acq.end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    
    # If email is provided, send email
    if len(email) != 0:
        if username is not None or app_password is not None:
            files = [plot_name]
            email_text = 'Greetings,\n\nYou are subscribed to the CHIME/FRB GBO outrigger data quality emails. Attached is the following:\n\t(1) A PDF containing a dynamic spectrum of the raw data coming out of the analog-to-digital converters attached to each feed. Specifically, this is the total dynamic spectrum summed over each feed.\n If you have any questions about the contents of this email, feel free to reach out to CHIME/FRB grad student Bridget Andersen at bridgetcandersen@gmail.com.\n Cheers,\n\n Bridget Andersen'
            subject = 'CHIME/FRB Outriggers Data Quality'
            send_email(subject, email_text, username, list(email), app_password, files=files)
        else:
            print("Cannot send email without Gmail username/password. Please specify the username/password at the beginning of raw_acq_cli.py.")

    # Remove plot if it shouldn't be saved...
    if not save_plot:
        os.remove(plot_name)

# Add commands to the CLI group
cli.add_command(plot_summed_spectrum)

if __name__ == "__main__":
    cli()
