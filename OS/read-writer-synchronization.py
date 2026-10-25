import threading
import time

# Shared state
shared_data = 0


class ReadWriteLock:
  """A custom Read-Write Lock implementation for Python threading."""

  def __init__(self):
    self._lock = threading.Lock()
    self._readers_reading = 0
    self._writers_waiting = 0
    self._writers_writing = False
    self._condition = threading.Condition(self._lock)

  def acquire_read(self):
    with self._condition:
      while self._writers_writing or self._writers_waiting > 0:
        self._condition.wait()
      self._readers_reading += 1

  def release_read(self):
    with self._condition:
      self._readers_reading -= 1
      if self._readers_reading == 0:
        self._condition.notify_all()

  def acquire_write(self):
    with self._condition:
      self._writers_waiting += 1
      while self._readers_reading > 0 or self._writers_writing:
        self._condition.wait()
      self._writers_waiting -= 1
      self._writers_writing = True

  def release_write(self):
    with self._condition:
      self._writers_writing = False
      self._condition.notify_all()


# Instantiate the lock
lock = ReadWriteLock()


def reader(reader_id):
  lock.acquire_read()
  try:
    print(f"Reader {reader_id} reads: {shared_data}")
  finally:
    lock.release_read()


def writer():
  global shared_data
  lock.acquire_write()
  try:
    shared_data = 100
    print(f"Writer updates data to: {shared_data}")
  finally:
    lock.release_write()


if __name__ == "__main__":
  # Start initial readers and the writer
  r1 = threading.Thread(target=reader, args=(1,))
  r2 = threading.Thread(target=reader, args=(2,))
  w1 = threading.Thread(target=writer)

  r1.start()
  r2.start()
  w1.start()

  # Wait for them to finish before starting r3 to match the expected timeline
  r1.join()
  r2.join()
  w1.join()

  # Now start Reader 3 after the writer has updated the data
  r3 = threading.Thread(target=reader, args=(3,))
  r3.start()
  r3.join()

  print(f"Final shared data: {shared_data}")
