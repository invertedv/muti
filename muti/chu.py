"""
Utilities that help with clickhouse

"""

def make_connection():
    """
      Connects to the local clickhouse server
      Returns the client
    """
    from clickhouse_driver import Client
    
    # establish connect, in this case to local server
    client = Client(host='localhost')
    
    # give queries more room to execute
    client.execute("SET max_memory_usage = 40000000000;")
    client.execute("SET max_bytes_before_external_group_by=20000000000;")
    return client


def run_query(query_or_file, client, is_file=False, return_df=False, replace_source=None, replace_dest=None):
    """

    Run a clickhouse query.
    
    :param query_or_file: either a query to run or a file containing a query to run
    :type query_or_file: str
    :param client: clickhouse Client
    :type client: clickhouse_driver.Client
    :param is_file: True means the first argument is a file name
    :type is_file: bool
    :param return_df: True means return the output as a DataFrame
    :type return_df: bool
    :param replace_source: text in the query to replace
    :type replace_source str or list of str
    :param replace_dest: text to put into the query
    :type replace_dest: str or list of str
    :return: pandas DataFrame if return_df=True
    :rtype DataFrame
    """

    query = query_or_file
    if is_file:
        query = ""
        f = open(query_or_file, 'r')
        while True:
            l = f.readline()
            if not l: break
            query += l
        f.close()
    if isinstance(replace_source, list):
        for j, src in enumerate(replace_source):
            query = query.replace(src, replace_dest[j])
    elif replace_source is not None:
        query = query.replace(replace_source, replace_dest)
    if return_df:
        df = client.query_dataframe(query)
        return df
    client.execute(query)


def import_flat_file(table_name, file_name, delim="|", format="CSV", options = ""):
    """
      Import a CSV into a clickhouse table

      table_name    str   name of table to insert into
      file_name     str   file to import
      delim         str   field delimiter in the input file
                          if null, then delim option is not used
      format        str   CSV or TabSeparated
    """
    
    import os
    cmd = "clickhouse-client " + options + " "
    if delim != "":
        cmd += '--format_csv_delimiter="' + delim + '" '
    cmd += ' --query "INSERT INTO ' + table_name + ' FORMAT ' + format + '"  < '
    cmd += file_name
    os.system(cmd)

