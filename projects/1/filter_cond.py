#
#
def filter_cond(line_dict):
    """Filter function
    Takes a dict with field names as argument
    Returns True if conditions are satisfied
    """
    if line_dict["if1"] == '':
        line_dict["if1"] = str('0')
    cond_match = (
       (int(line_dict["if1"]) > 20) and (int(line_dict["if1"]) < 40)
    ) 
    return True if cond_match else False
