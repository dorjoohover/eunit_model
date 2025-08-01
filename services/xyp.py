# -- coding: utf-8 --
from zeep import Client
from zeep.transports import Transport
from requests import Session
import urllib3
import os
import time
from base64 import b64encode
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5
from Crypto.PublicKey import RSA




class XypSign:
    def __init__(self, KeyPath):
        self.KeyPath = KeyPath

    def __GetPrivKey(self):
        with open(self.KeyPath, "rb") as keyfile:
            return RSA.importKey(keyfile.read())

    def __toBeSigned(self, accessToken):
        return {
            'accessToken': accessToken,
            'timeStamp': self.__timestamp(),
        }

    def __buildParam(self, toBeSigned):
        return toBeSigned['accessToken'] + '.' + toBeSigned['timeStamp']

    def sign(self, accessToken):
        toBeSigned = self.__toBeSigned(accessToken)
        digest = SHA256.new()
        digest.update(self.__buildParam(toBeSigned).encode('utf8'))
        pkey = self.__GetPrivKey()
        dd = b64encode(PKCS1_v1_5.new(pkey).sign(digest))
        return toBeSigned, dd

    def __timestamp(self):
        return str(int(time.time()))


class Service:
    def __init__(self, wsdl, accesstoken, pkey_path=None):
        self.__accessToken = accesstoken
        self.__toBeSigned, self.__signature = XypSign(
            pkey_path).sign(self.__accessToken)
        urllib3.disable_warnings()
        session = Session()
        session.verify = False
        transport = Transport(session=session)

        self.client = Client(wsdl, transport=transport)
        self.client.transport.session.headers.update({
            'accessToken': self.__accessToken,
            'timeStamp': self.__toBeSigned['timeStamp'],
            'signature': self.__signature
        })

    def dump(self, operation, params=None):
        try:
            if params:
                response = self.client.service[operation](params)
                return response
            else:
                return self.client.service[operation]()
        except Exception as e:
            print(operation, str(e))


