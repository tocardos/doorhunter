# config.py
import os
from quart import Quart
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, select
from sqlalchemy import desc
import asyncio
from functools import wraps
import logging
from datetime import datetime
# 4G settings
from huawei_lte_api.Connection import Connection
from huawei_lte_api.Client import Client
from huawei_lte_api.enums.client import ResponseEnum
import usb.core
import usb.util
# for wiifi settings
import NetworkManager as nm
import dbus.mainloop.glib
from gi.repository import GLib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOTSPOT_UUID = "fa1185cf-e8a2-4ad2-9b0e-bc1704f71b36"


class WiFi:
    def __init__(self):
        self.uuid = None
        # Initialize D-Bus main loop
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.loop = GLib.MainLoop()
        
    async def list_connections(self):
        # Run the D-Bus operation in a thread pool
        connections = await asyncio.get_event_loop().run_in_executor(
            None, self._list_connections_sync)
        return connections

    def _list_connections_sync(self):
        """Synchronous method to list connections"""
        connections = []
        try:
            for conn in nm.Settings.ListConnections():
                settings = conn.GetSettings()
                connections.append({
                    'name': settings['connection']['id'],
                    'uuid': settings['connection']['uuid']
                })
        except Exception as e:
            logger.error(f"Error listing connections: {str(e)}")
        return connections

    async def get_active_connections(self):
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_active_connections_sync)

    def _get_active_connections_sync(self):
        """Synchronous method to get active connections"""
        active_connections = []
        try:
            for active in nm.NetworkManager.ActiveConnections:
                conn_settings = active.Connection.GetSettings()
                active_connections.append({
                    'name': conn_settings['connection']['id'],
                    'uuid': conn_settings['connection']['uuid'],
                    'type': conn_settings['connection']['type']
                })
        except Exception as e:
            logger.error(f"Error getting active connections: {str(e)}")
        return active_connections
    async def get_connection_by_uuid(self,uuid):
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_active_connections_sync(uuid))
        
    def _get_connection_by_uuid_sync(self,uuid):
        try:
            return nm.Settings.GetConnectionByUuid(uuid)
        except Exception as e:
            logger.error(f"Error in getting connection by uuid: {str(e)}")
            return None
    async def get_active_connection_by_uuid(self,uuid):
        return await asyncio.get_event_loop().run_in_executor(
            None,self._get_active_connection_by_uuid_sync(uuid))
    def _get_active_connection_by_uuid_sync(self,uuid):
        for conn in nm.NetworkManager.ActiveConnections:
            if conn.Connection.GetSettings()['connection']['uuid'] == uuid:
                return conn
        return None
    async def set_active_connection(self,connection):
        return await asyncio.get_event_loop().run_in_executor(
            None,self._set_active_connection_sync(connection)
        )
    def _set_active_connection_sync(self,connection):
        nm.NetworkManager.ActivateConnection(connection, "/", "/")
        
    

# Create a global instance of WiFi
wifi_con = WiFi()

async def connection_status_wifi():
    try:
        connections = await wifi_con.list_connections()
        active_connections = await wifi_con.get_active_connections()
        #print(f"active connections {active_connections}")
        #print(f"connections {connections}")
        return {
            "all_connections": connections,
            "active_connections": active_connections
        }
    except Exception as e:
        logger.error(f"Error in connection status wifi: {str(e)}")
        return {
            'all_connections': [],
            'active_connections': []
        }
async def get_hotspot_status():
    try:
       active_conn =   wifi_con.get_active_connection_by_uuid(HOTSPOT_UUID)
       #print(f"active conn {active_conn}")
       return bool(active_conn)
    except Exception as e:
       logger.error(f"error trying to get hotspot status {str(e)}")



#---------------------------------------------------------------
#
# 4G dongle API
#
#--------------------------------------------------------------
class Dongle:
    def __init__(self, url='http://192.168.8.1/', vendor_id=0x12d1): # huawei id
        self.url = url
        self._connection_status = False
        self._last_check = None
        self.vendor_id = vendor_id
    
    def is_dongle_connected(self):
        try:
            # Find all USB devices with the specified vendor ID
            devices = list(usb.core.find(idVendor=self.vendor_id, find_all=True))
            return len(devices) > 0
        except Exception as e:
            logger.error(f"Error checking USB devices: {str(e)}")
            return False
        
    async def get_connection_status(self):
        def _get_status():
            try:
                logger.info("Checking 4G connection status...")
                if not self.is_dongle_connected():
                    logger.warning("4G USB dongle not found")
                    return False
                with Connection(self.url) as connection:
                    client = Client(connection)
                    status = client.monitoring.status()
                    test = status["ConnectionStatus"]
                    print(f"4G status {test}")
                    return status["ConnectionStatus"] == "901"
            except Exception as e:
                logger.error(f"Error checking 4G connection: {str(e)}")
                return False

        # Run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_status)

    async def send_sms(self, phone_number, message):
        def _send_sms():
            try:
                logger.info(f"Attempting to send SMS to {phone_number}")
                with Connection(self.url) as connection:
                    client = Client(connection)
                    return client.sms.send_sms([phone_number], message) == ResponseEnum.OK.value
            except Exception as e:
                logger.error(f"Error sending SMS: {str(e)}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _send_sms)

# Create a global dongle instance
dongle = Dongle()


async def connection_status_processor():
    try:
        status = await dongle.get_connection_status()
        print("connection status processor")
        return {'connection_status': status}
    except Exception as e:
        logger.error(f"Error in connection status processor: {str(e)}")
        return {'connection_status': False}

#------------------------------------------------------------
#
# DATABASE DEFINITION
#
#------------------------------------------------------------

# Adjust the path for the database in the static folder
STATIC_FOLDER = 'static'
DB_NAME = 'database.db'
DATABASE_PATH = os.path.join(STATIC_FOLDER, DB_NAME)

# SQLite URL for async
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False}
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Create declarative base
Base = declarative_base()

# Database Models
class Contact(Base):
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    alias = Column(String(80), unique=True, nullable=False)
    first_name = Column(String(80), nullable=False)
    last_name = Column(String(80), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    mail_send = Column(Boolean, default=False)
    phone = Column(String(20), unique=True, nullable=False)
    sms_send = Column(Boolean, default=False)

class SignalGroup(Base):
    __tablename__ = 'signal_groups'
    
    id = Column(Integer, primary_key=True)
    alias = Column(String(80), unique=True, nullable=False)
    group_name = Column(String(80), nullable=False)
    participants = Column(Text, nullable=False)

class Recording(Base):
    __tablename__ = 'recordings'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(120), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    reminder = Column(Boolean, default=False)

# Database utility functions
async def init_db():
    # Ensure the static directory exists
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with async_session() as session:
        yield session

# DB operations class
class DatabaseOperations:
    @staticmethod
    async def add_contact(session, contact_data):
        contact = Contact(**contact_data)
        session.add(contact)
        await session.commit()
        return contact

    @staticmethod
    async def get_contact_by_alias(session, alias):
        result = await session.execute(
            select(Contact).filter(Contact.alias == alias)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def add_recording(session, filename):
        recording_data = {
            'filename': filename,
            'timestamp': datetime.now()
        }
        recording = Recording(**recording_data)
        session.add(recording)
        await session.commit()
        return recording
    @staticmethod
    async def get_recordings(session):
        
        result = await session.execute(
            select(Recording).order_by(desc(Recording.timestamp))
        )
        recordings = result.scalars().all()
        return recordings
    @staticmethod
    async def get_all_contacts(session):
        result = await session.execute(select(Contact))
        return result.scalars().all()
    @staticmethod
    async def get_contact_by_id(session, contact_id):
        result = await session.execute(
            select(Contact).filter(Contact.id == contact_id)
        )
        return result.scalar_one_or_none()
    @staticmethod
    async def update_contact(session, contact, **kwargs):
        for key, value in kwargs.items():
            setattr(contact, key, value)
        await session.commit()
        return contact

    @staticmethod
    async def delete_contact(session, contact):
        await session.delete(contact)
        await session.commit()
    @staticmethod
    async def send_sms_to_contacts(session, message, sms_enabled_only=True):
        # Get all contacts or only those with sms_send enabled
        query = select(Contact)
        if sms_enabled_only:
            query = query.filter(Contact.sms_send == True)
        
        result = await session.execute(query)
        contacts = result.scalars().all()
        
        success_count = 0
        failed_contacts = []
        for contact in contacts:
            try:
                if await dongle.send_sms(contact.phone, message):
                    success_count += 1
                    logger.debug(f"Successfully sent SMS to {contact.phone}")
                else:
                    failed_contacts.append(contact.phone)
                    logger.warning(f"Failed to send SMS to {contact.phone}")
            except Exception as e:
                failed_contacts.append(contact.phone)
                logger.error(f"Error sending SMS to {contact.phone}: {str(e)}")
        
        
        return success_count, len(contacts)