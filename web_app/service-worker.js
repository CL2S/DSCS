/* ==========================================================================
   Sepsis Prediction System - Service Worker
   Provides offline capabilities and resource caching
   ========================================================================== */

const CACHE_NAME = 'sepsis-prediction-v2';
const CACHE_VERSION = '2.0.0';

// Assets to cache immediately on install
const PRECACHE_ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.js'
];

// External resources to cache
const EXTERNAL_RESOURCES = [
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Source+Code+Pro:wght@400;500;600&display=swap',
  'https://unpkg.com/@phosphor-icons/web@2.1.1/src/regular/style.css',
  'https://cdn.plot.ly/plotly-2.27.0.min.js',
  'https://unpkg.com/vue@3/dist/vue.global.js',
  'https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js'
];

// Install event - precache critical assets
self.addEventListener('install', event => {
  console.log(`[Service Worker] Installing version ${CACHE_VERSION}`);

  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Precaching assets');
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => {
        console.log('[Service Worker] Precaching completed');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('[Service Worker] Precaching failed:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log(`[Service Worker] Activating version ${CACHE_VERSION}`);

  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== CACHE_NAME) {
              console.log(`[Service Worker] Deleting old cache: ${cacheName}`);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[Service Worker] Claiming clients');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  // Skip non-GET requests and browser extensions
  if (event.request.method !== 'GET' ||
      event.request.url.startsWith('chrome-extension://') ||
      event.request.url.includes('extension')) {
    return;
  }

  // Handle API requests differently
  if (event.request.url.includes('/api/')) {
    handleApiRequest(event);
    return;
  }

  // Handle external resources
  if (EXTERNAL_RESOURCES.some(url => event.request.url.includes(url))) {
    handleExternalResource(event);
    return;
  }

  // Handle local resources with cache-first strategy
  handleLocalResource(event);
});

// Strategy for API requests: network-first, fallback to cache
function handleApiRequest(event) {
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Don't cache error responses
        if (!response.ok) {
          return response;
        }

        // Clone the response to cache it
        const responseToCache = response.clone();

        caches.open(CACHE_NAME)
          .then(cache => {
            // Cache API responses for offline use
            // Only cache GET requests and successful responses
            if (event.request.method === 'GET') {
              cache.put(event.request, responseToCache);
            }
          })
          .catch(error => {
            console.error('[Service Worker] Failed to cache API response:', error);
          });

        return response;
      })
      .catch(() => {
        // Network failed, try cache
        return caches.match(event.request)
          .then(cachedResponse => {
            if (cachedResponse) {
              console.log('[Service Worker] Serving API from cache:', event.request.url);
              return cachedResponse;
            }

            // Return offline response for API calls
            return new Response(
              JSON.stringify({
                error: 'offline',
                message: 'You are offline. Please check your connection.',
                timestamp: new Date().toISOString()
              }),
              {
                status: 503,
                statusText: 'Service Unavailable',
                headers: { 'Content-Type': 'application/json' }
              }
            );
          });
      })
  );
}

// Strategy for external resources: cache-first, update in background
function handleExternalResource(event) {
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        // Return cached response if available
        if (cachedResponse) {
          // Update cache in background
          fetchAndCache(event.request);
          return cachedResponse;
        }

        // Not in cache, fetch from network
        return fetchAndCache(event.request);
      })
  );
}

// Strategy for local resources: cache-first
function handleLocalResource(event) {
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        if (cachedResponse) {
          return cachedResponse;
        }

        return fetch(event.request)
          .then(response => {
            // Check if we received a valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            // Clone the response to cache it
            const responseToCache = response.clone();

            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseToCache);
              });

            return response;
          })
          .catch(() => {
            // If both cache and network fail, return offline page for HTML requests
            if (event.request.headers.get('accept').includes('text/html')) {
              return caches.match('./index.html');
            }

            // For other resources, return appropriate offline response
            if (event.request.url.endsWith('.css')) {
              return new Response(
                '/* Offline - styles not available */',
                { headers: { 'Content-Type': 'text/css' } }
              );
            }

            if (event.request.url.endsWith('.js')) {
              return new Response(
                '// Offline - JavaScript not available',
                { headers: { 'Content-Type': 'application/javascript' } }
              );
            }

            return new Response('Offline', { status: 503 });
          });
      })
  );
}

// Helper function to fetch and cache a resource
function fetchAndCache(request) {
  return fetch(request)
    .then(response => {
      // Don't cache if not a valid response
      if (!response || response.status !== 200) {
        return response;
      }

      // Clone response to cache
      const responseToCache = response.clone();

      caches.open(CACHE_NAME)
        .then(cache => {
          cache.put(request, responseToCache);
        });

      return response;
    })
    .catch(error => {
      console.error('[Service Worker] Failed to fetch and cache:', request.url, error);
      throw error;
    });
}

// Background sync for offline predictions
self.addEventListener('sync', event => {
  if (event.tag === 'sync-predictions') {
    console.log('[Service Worker] Background sync triggered');
    event.waitUntil(syncPredictions());
  }
});

// Function to sync pending predictions
function syncPredictions() {
  // This would sync any predictions made while offline
  // In a real implementation, you would store predictions in IndexedDB
  // and sync them when back online
  console.log('[Service Worker] Syncing pending predictions...');
  return Promise.resolve();
}

// Push notification support
self.addEventListener('push', event => {
  const options = {
    body: event.data?.text() || 'New prediction result available',
    icon: '/icon-192.png',
    badge: '/badge-72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 'prediction-result'
    },
    actions: [
      {
        action: 'view',
        title: 'View Results'
      },
      {
        action: 'close',
        title: 'Close'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Sepsis Prediction System', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  console.log('[Service Worker] Notification click received');

  event.notification.close();

  if (event.action === 'view') {
    // Open the application
    event.waitUntil(
      clients.matchAll({ type: 'window' })
        .then(windowClients => {
          if (windowClients.length > 0) {
            return windowClients[0].focus();
          }
          return clients.openWindow('/');
        })
    );
  }
});

// Periodic background updates
self.addEventListener('periodicsync', event => {
  if (event.tag === 'update-resources') {
    console.log('[Service Worker] Periodic sync triggered');
    event.waitUntil(updateCachedResources());
  }
});

// Update cached resources in background
function updateCachedResources() {
  return caches.open(CACHE_NAME)
    .then(cache => {
      const updatePromises = PRECACHE_ASSETS.map(asset => {
        return fetch(asset, { cache: 'reload' })
          .then(response => {
            if (response.ok) {
              return cache.put(asset, response);
            }
          })
          .catch(error => {
            console.error(`[Service Worker] Failed to update ${asset}:`, error);
          });
      });

      return Promise.all(updatePromises);
    })
    .then(() => {
      console.log('[Service Worker] Resource update completed');
    });
}

// Error handling
self.addEventListener('error', event => {
  console.error('[Service Worker] Error:', event.error);
});

self.addEventListener('unhandledrejection', event => {
  console.error('[Service Worker] Unhandled rejection:', event.reason);
});