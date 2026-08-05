import urllib.request
import json

try:
    url = "https://api.github.com/repos/tamimystic/Chokkhu-PyPi-Package/actions/runs?per_page=5"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req)
    data = json.loads(res.read())
    
    # Get latest failed run for CI
    run_id = None
    for run in data['workflow_runs']:
        if run['name'] == 'Chokkhu CI' and run['conclusion'] == 'failure':
            run_id = run['id']
            break
            
    if not run_id:
        print("No failed CI runs found recently.")
    else:
        print(f"Failed Run ID: {run_id}")
        jobs_url = f"https://api.github.com/repos/tamimystic/Chokkhu-PyPi-Package/actions/runs/{run_id}/jobs"
        req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        jobs_data = json.loads(res.read())
        
        for job in jobs_data['jobs']:
            if job['conclusion'] == 'failure':
                print(f"\n--- FAILED JOB: {job['name']} ---")
                log_url = f"https://api.github.com/repos/tamimystic/Chokkhu-PyPi-Package/actions/jobs/{job['id']}/logs"
                try:
                    log_req = urllib.request.Request(log_url, headers={'User-Agent': 'Mozilla/5.0'})
                    log_res = urllib.request.urlopen(log_req)
                    logs = log_res.read().decode('utf-8')
                    print("\n".join(logs.splitlines()[-100:]))
                except Exception as e:
                    print(f"Failed to fetch log: {e}")

except Exception as e:
    print(f"Error: {e}")

