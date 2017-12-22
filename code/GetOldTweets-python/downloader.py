from subprocess import call

month = '2017-09-'
location = 'Las Vegas'
begin_day = 30
end_day = 31
max_tweets_per_day = 2000
for i in xrange(begin_day, end_day):
    frmday = str(i)
    if i < 10:
        frmday = "0" + frmday
    today = str(i+1)
    if i+1 < 10:
        today = "0" + today
    frm = month+frmday
    until = month+today
    output_dir = 'data_'+location.replace(" ", "_")+ '/'
    call(['mkdir', output_dir])
    if max_tweets_per_day == -1:
        cmd = ["python", "Exporter.py", "--since", frm, "--until", until, "--near", location, "--output", output_dir + frm +".csv"]
    else:
        cmd = ["python", "Exporter.py", "--since", frm, "--until", until, "--near", location, "--maxtweets", str(max_tweets_per_day), "--output", output_dir + frm +".csv"]
    call(cmd)
