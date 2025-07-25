
with open('app.py', 'r') as file:
    content = file.read()

if '@app.route(\'/lawyer\')' not in content:
    # Find the position to insert the new route
    landing_route = '@app.route(\'/landing\')'
    landing_def = 'def landing():'
    landing_return = 'return render_template(\
landing.html\)'
    
    # Find the end of the landing route function
    landing_start = content.find(landing_route)
    if landing_start != -1:
        landing_end = content.find('\\n\\n', content.find(landing_return, landing_start))
        if landing_end != -1:
            # Insert the lawyer route after the landing route
            new_route = '\\n\\n@app.route(\'/lawyer\')\\ndef lawyer():\\n    return render_template(\lawyer.html\)'
            modified_content = content[:landing_end] + new_route + content[landing_end:]
            
            with open('app.py', 'w') as file:
                file.write(modified_content)
            
            print('Lawyer route added successfully!')
        else:
            print('Could not find the end of the landing route function')
    else:
        print('Could not find the landing route')
else:
    print('Lawyer route already exists')

