import re
import os
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from gensim.models import KeyedVectors


def total_special_characters(url_remove_protocol):
    characters = ['-', '&', '@', '.', '~', '%', '#', '_']
    total_special = {}
    for c in characters:
        total_special[c] = 0
    for c in url_remove_protocol:
        if c in characters:
            total_special[c] += 1
    return total_special


def numDots(total_special):
    return total_special['.']


def numDash(total_special):
    return total_special['-']


def atSymbol(total_special):
    if (total_special['@'] > 0):
        return 1
    return 0
    # return total_special['@']


def tildeSymbol(total_special):
    if total_special['~'] > 0:
        return 1
    return 0
    # return total_special['~']


def numUnderscore(total_special):
    return total_special['_']


def numPercent(total_special):
    return total_special['%']


def numAmpersand(total_special):
    return total_special['&']


def numHash(total_special):
    return total_special['#']


def numNumericChars(url_remove_protocol):
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    count = 0
    for c in url_remove_protocol:
        if c in characters:
            count += 1
    return count


def get_TLD():
    f = open("Dataset/File Path/list_tld.txt", "r")
    data = []
    for x in f:
        data.append(x[:len(x)-1].lower())
    return data


def domainToken(hostname):
    domain_tokens = hostname.split('.')
    domain_tokens = [x for x in domain_tokens if x != '']
    return domain_tokens


def subdomainLevel(domain_tokens):
    count = 0
    for i in range(len(domain_tokens)):
        if domain_tokens[len(domain_tokens)-1-i] in TLDs:
            count += 1
        else:
            break
    return len(domain_tokens)-count-1


def pathLevel(paths):
    paths_tokens = [re.sub('/', '', x) for x in paths]
    return len(paths_tokens)


def urlLength(url_remove_protocol):
    return len(url_remove_protocol)


def numDashInHostname(hostname):
    characters = ['-']
    count = 0
    for c in hostname:
        if c in characters:
            count += 1
    return count


def noHttps(url):
    a = url[:5]
    a = a.lower()
    if a == "https":
        return 1
    return 0


def ip_address(domain_tokens):
    for i in domain_tokens:
        if i.isdigit() == False:
            return 0
    else:
        return 1


def get_domain_and_subdomain(domain_tokens):
    count = 0
    for i in range(len(domain_tokens)):
        if domain_tokens[len(domain_tokens)-1-i] in TLDs:
            count += 1
        else:
            break
    sub_domain = domain_tokens[:len(domain_tokens)-1-count]
    domain = domain_tokens[len(domain_tokens)-1-count:len(domain_tokens)]
    sub_domain_str = ""
    for i in range(len(sub_domain)):
        if i == len(sub_domain)-1:
            sub_domain_str += sub_domain[i]
        else:
            sub_domain_str += sub_domain[i] + "."
    domain_str = ""
    for i in range(len(domain)):
        if i == len(domain)-1:
            domain_str += domain[i]
        else:
            domain_str += domain[i] + "."
    return sub_domain_str, domain_str


def domainInSubdomains(domain_tokens):
    sub_domain_str, domain_str = get_domain_and_subdomain(domain_tokens)
    if sub_domain_str.find(domain_str) >= 0:
        return 1
    return 0


def domainInPaths(paths, domain_tokens):
    _, domain_str = get_domain_and_subdomain(domain_tokens)
    paths_tokens = [re.sub('/', '', x) for x in paths]
    if domain_str in paths_tokens:
        return 1
    return 0


def httpsInHostname(hostname):
    if hostname.find("https") >= 0:
        return 1
    return 0


def hostnameLength(hostname):
    return len(hostname)


def pathLength(paths):
    count = 0
    for x in paths:
        count += len(x)
    return count


def doubleSlashInPath(paths):
    for x in paths:
        if x.find("//") > 0:
            return 1
    return 0


def numSensitiveWords(url_remove_protocol):
    Suspicious_Words = ['secure', 'account', 'update', 'banking', 'login', 'click',
                        'confirm', 'password', 'verify', 'signin', 'ebayisapi', 'lucky', 'bonus']
    count = 0
    for x in Suspicious_Words:
        count += len(url_remove_protocol.split(x))-1
    return count


def frequentDomainNameMismatch(domain_tokens):
    if len(domain_tokens) == 0:
        return 0
    TLD = domain_tokens[-1]
    if TLD in TLDs:
        return 1
    return 0


def url_length_RT(url_remove_protocol):
    if len(url_remove_protocol) < 54:
        return 1
    elif len(url_remove_protocol) >= 54 and len(url_remove_protocol) <= 75:
        return 0
    else:
        return -1


def num_of_query(paths):
    paths_tokens = [re.sub('/', '', x) for x in paths]
    count = 0
    for x in paths_tokens:
        o = urlparse(x)
        if len(o.query) > 0:
            count += 1
    return count


def size_of_query(paths):
    paths_tokens = [re.sub('/', '', x) for x in paths]
    size = 0
    for x in paths_tokens:
        o = urlparse(x)
        size += len(o.query)
    return size


def subdomainLevelRT(domain_tokens):
    sub_domain_str, _ = get_domain_and_subdomain(domain_tokens)
    s = sub_domain_str.split(".")
    if len(s) == 1:
        return 1
    if len(s) == 2:
        return 0
    return -1


def load_word_vectors(path):
    return KeyedVectors.load(path)


def generate_path(path_list):
    WORD_VECTORS_PATH = os.path.join(
        'Model/File Path/Word Vector', 'fasttext_v1.model')
    if os.path.isfile(WORD_VECTORS_PATH):
        word_vectors = load_word_vectors(WORD_VECTORS_PATH)

    path_df = pd.DataFrame(path_list).rename(columns={0: 'path_origin'})
    path_df['text_tokenized'] = path_df['path_origin'].str.lower().str.split('\\')
    path_df['text_vect_mean'] = path_df['text_tokenized'].apply(
        lambda x: np.array([word_vectors[token] for token in x]).mean(axis=0))

    return path_df.iloc[:, -1]


def generate_event(df):
    WORD_VECTORS_PATH = os.path.join(
        'Model/Event/Word Vector', 'event.model')
    if os.path.isfile(WORD_VECTORS_PATH):
        word_vectors = load_word_vectors(WORD_VECTORS_PATH)

    df['text_tokenized'] = df['Info'].str.split('\\')
    df['text_vect_mean'] = df['text_tokenized'].apply(
        lambda x: np.array([word_vectors[token] for token in x]).mean(axis=0))

    return df


def generate_url(url_list):
    url_df = pd.DataFrame(url_list).rename(columns={0: 'URL_origin'})
    url_df['URL'] = url_df['URL_origin'].apply(
        lambda x: re.sub(r'^http(s*)://', '', x))
    url_df['hostname'] = url_df['URL'].apply(
        lambda x: re.match(r'^[^/]*', x).group(0))
    url_df['domain_tokens'] = url_df['hostname'].apply(
        lambda x: domainToken(x))
    url_df['paths'] = url_df['URL'].apply(lambda x: re.findall(r'/[^/]*', x))
    url_df['total_special'] = url_df['URL'].apply(
        lambda x: total_special_characters(x))

    url_df['numDots'] = url_df['total_special'].apply(lambda x: numDots(x))
    url_df['subdomainLevel'] = url_df['domain_tokens'].apply(
        lambda x: subdomainLevel(x))
    url_df['pathLevel'] = url_df['paths'].apply(lambda x: pathLevel(x))
    url_df['urlLength'] = url_df['URL'].apply(lambda x: urlLength(x))
    url_df['numDash'] = url_df['total_special'].apply(lambda x: numDash(x))
    url_df['numDashInHostname'] = url_df['hostname'].apply(
        lambda x: numDashInHostname(x))
    url_df['atSymbol'] = url_df['total_special'].apply(lambda x: atSymbol(x))
    url_df['tildeSymbol'] = url_df['total_special'].apply(
        lambda x: tildeSymbol(x))
    url_df['numUnderscore'] = url_df['total_special'].apply(
        lambda x: numUnderscore(x))
    url_df['numPercent'] = url_df['total_special'].apply(
        lambda x: numPercent(x))
    url_df['count_query'] = url_df['paths'].apply(lambda x: num_of_query(x))
    url_df['numAmpersand'] = url_df['total_special'].apply(
        lambda x: numAmpersand(x))
    url_df['numHash'] = url_df['total_special'].apply(lambda x: numHash(x))
    url_df['numNumericChars'] = url_df['URL'].apply(
        lambda x: numNumericChars(x))
    url_df['noHttps'] = url_df['URL_origin'].apply(lambda x: noHttps(x))
    url_df['ip_address'] = url_df['domain_tokens'].apply(
        lambda x: ip_address(x))
    url_df['domainInSubdomains'] = url_df['domain_tokens'].apply(
        lambda x: domainInSubdomains(x))
    url_df['domainInPaths'] = url_df.apply(
        lambda x: domainInPaths(x['paths'], x['domain_tokens']), axis=1)
    url_df['httpsInHostname'] = url_df['hostname'].apply(
        lambda x: httpsInHostname(x))
    url_df['hostnameLength'] = url_df['hostname'].apply(
        lambda x: hostnameLength(x))
    url_df['size_query'] = url_df['paths'].apply(lambda x: size_of_query(x))
    url_df['pathLength'] = url_df['paths'].apply(lambda x: pathLength(x))
    url_df['doubleSlashInPath'] = url_df['paths'].apply(
        lambda x: doubleSlashInPath(x))
    url_df['numSensitiveWords'] = url_df['URL'].apply(
        lambda x: numSensitiveWords(x))
    url_df['frequentDomainNameMismatch'] = url_df['domain_tokens'].apply(
        lambda x: frequentDomainNameMismatch(x))
    url_df['url_length_RT'] = url_df['URL'].apply(lambda x: url_length_RT(x))
    url_df['subdomainLevelRT'] = url_df['domain_tokens'].apply(
        lambda x: subdomainLevelRT(x))
    return url_df.iloc[:, 6:]
