"""
Utilities that help with clickhouse

"""
import clickhouse_driver


def make_connection(threads=4):
    """
      Connects to the local clickhouse server
      Returns the client
    """
    from clickhouse_driver import Client
    
    # establish connect, in this case to local server
    client = Client(host='localhost')
    
    # Consider using:
    # client = Client(host='localhost', settings={'use_numpy': True})

    # give queries more room to execute
    client.execute("SET max_memory_usage = 40000000000;")
    client.execute("SET max_bytes_before_external_group_by=20000000000;")
    client.execute("SET max_threads=" + str(threads))
    return client


def run_query(query_or_file: str, client: clickhouse_driver.Client, is_file=False,
              return_df=False, replace_source=None, replace_dest=None):
    """

    Run a clickhouse query.
    
    :param query_or_file: either a query to run or a file containing a query to run
    :param client: clickhouse Client
    :param is_file: True means the first argument is a file name
    :param return_df: True means return the output as a DataFrame
    :param replace_source: text in the query to replace
    :param replace_dest: text to put into the query
    :return: pandas DataFrame if return_df=True
    :rtype DataFrame
    """

    query = query_or_file
    if is_file:
        query = ""
        f = open(query_or_file, 'r')
        while True:
            l = f.readline()
            if not l:
                break
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


def import_flat_file(table_name: str, file_name: str, delim="|", format="CSV", options=""):
    """
    Import a flat file into ClickHouse
    
    :param table_name: table to import into (fully qualified with db name)
    :param file_name: file to read from
    :param delim: delimiter in the file
    :param format: file format
    :param options: other clickhouse options
    :return:
    """
    
    import os
    cmd = "clickhouse-client " + options + " "
    if delim != "":
        cmd += '--format_csv_delimiter="' + delim + '" '
    cmd += ' --query "INSERT INTO ' + table_name + ' FORMAT ' + format + '"  < '
    cmd += file_name
    os.system(cmd)
