const CACHE_NAME = 'chiller-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png'
];

// Install service worker
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(ASSETS))
  );
});

// Activate and clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key !== CACHE_NAME)
           .map(key => caches.delete(key))
      );
    })
  );
});

// Fetch strategy: Cache first, then network
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then(cached => {
        return cached || fetch(event.request)
          .then(response => {
            const cache = caches.open(CACHE_NAME);
            cache.then(cache => cache.put(event.request, response.clone()));
            return response;
          });
      })
      .catch(() => {
        // Return offline fallback if no cache or network
        return caches.match('/');
      })
  );
});