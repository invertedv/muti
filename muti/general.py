"""
muti that are helpful in general model building

"""
def get_unique_levels(feature, client, db, table):
    """
    Retrieves the unique levels of the column 'feature' in the table 'table' of database db.

    :param feature: column name in db.table to get unique levels
    :type str
    :param client: clickhouse client connector
    :type clickhouse_driver.Client
    :param db: database name
    :type str
    :param table: table name
    :type str
    :return: list of unique levels
    """
    qry = 'SELECT DISTINCT ' + feature + ' FROM ' + db + '.' + table + ' ORDER BY ' + feature
    uf = client.execute(qry)
    return [u[0] for u in uf]
